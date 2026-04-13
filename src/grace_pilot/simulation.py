import os 
import subprocess
import re
import yaml 
import shutil

from .machine import machine 

CHAIN_STATES = ("PENDING", "RUNNING", "SUSPENDED", "CONFIGURING", "COMPLETING", "SUSPENDING")


def fill_submit_template(template_file,output_file,replacements):

    with open(template_file, "r") as f:
        content = f.read()

    for key, val in replacements.items():
        placeholder = f"@{key}@"
        content = content.replace(placeholder, str(val))

    with open(output_file, "w") as f:
        f.write(content)


class simulation:

    def __init__(self, name, simdir, machine=None, submitscript=None, exe=None, parfile=None, env=None):
        self._name = name
        self._dir = simdir
        self._cdir = os.path.join(self._dir, "config")
        self._machine = machine
        self._exe = exe
        self._pfile = parfile
        self._subscript = submitscript
        self._env = env

        if not os.path.isdir(simdir):
            self._init_directory_structure()
        
        self._parse_dir()

    
    def submit(self, sub_args):
        rid = int(self._lastjob["restart_id"]) + 1
        prev_jobid = int(self._lastjob["job_id"])

        sub_args["JOBNAME"] = self._name + f"-{rid:04d}"

        simdir = os.path.join(self._dir, f"restart_{rid:04d}")
        os.makedirs(simdir)

        try:
            _ = self._copyfile(self._exe, simdir)
            pfile = self._copyfile(self._pfile, simdir)
            sub_args["PARAMETER_FILE"] = pfile
            sub_args["JOBDIR"] = simdir
            if self._env is not None:
                envfile = self._copyfile(self._env, simdir)
                sub_args["ENV_FILE"] = envfile
            sfile = self._edit_submit_script(simdir, sub_args)

            # check whether chaining is necessary
            need_chain = False
            if rid > 0:
                status = self._machine.scheduler.getstatus(prev_jobid)
                need_chain = status in CHAIN_STATES

            if need_chain:
                jobid = self._machine.scheduler.chain_submission(sfile, prev_jobid)
                print(f"Submitted simulation {self._name}, restart id {rid}, job id {jobid} with dependency {prev_jobid}")
            else:
                jobid = self._machine.scheduler.submit(sfile)
                print(f"Submitted simulation {self._name}, restart id {rid}, job id {jobid}")
        except Exception:
            # Clean up the restart directory so the next submit attempt
            # can recreate it cleanly with the same restart id.
            shutil.rmtree(simdir, ignore_errors=True)
            raise

        self._update_status(rid, jobid)
    


        
    def _update_status(self, rid, jid):
        self._lastjob["job_id"] = jid
        self._lastjob["restart_id"] = rid
        with open(self._info_file, "w") as f:
            yaml.safe_dump(self._lastjob, f)

    
    def _edit_submit_script(self, simdir, args):
        spath, sname = os.path.split(self._subscript)
        subfile = os.path.join(simdir, sname)
        self._machine.check_submit_arguments_and_set_defaults(args)
        fill_submit_template(self._subscript, subfile, args)
        return subfile

    def _copyfile(self,srcfile,dstpath):
        pth,nm = os.path.split(srcfile)
        dstfile = os.path.join(dstpath,nm)
        shutil.copy2(srcfile,dstfile)
        return dstfile

    def _init_directory_structure(self):

        cdir = os.path.join(self._dir,"config")
        os.makedirs(cdir)

        if self._exe is None:
            raise ValueError("Executable must be specified when creating a new simulation")
        if self._pfile is None:
            raise ValueError("Parameter file must be specified when creating a new simulation")
        if self._machine is None: 
            raise ValueError("Machine specs must be specified when creating a new simulation")
        if self._subscript is None:
            raise ValueError("Submit script must be specified when creating a new simulation")
        if self._env is None:
            raise ValueError("Environment file must be specified when creating a new simulation")

        self._machine.dump_config(os.path.join(cdir,"machine.yaml"))
        
        shutil.copy2(self._exe, os.path.join(cdir, 'grace'))

        _, pname = os.path.split(self._pfile)
        os.makedirs(os.path.join(cdir, 'parfile'))
        shutil.copy2(self._pfile, os.path.join(cdir, 'parfile', pname))

        os.makedirs(os.path.join(cdir, 'submission'))
        shutil.copy2(self._subscript, os.path.join(cdir, 'submission', "submission_script.x"))

        _, ename = os.path.split(self._env)
        os.makedirs(os.path.join(cdir, 'env'))
        shutil.copy2(self._env, os.path.join(cdir, 'env', ename))

        self._info_file = os.path.join(cdir, "status.yaml")
        self._lastjob = {"parfile": pname}

        self._update_status(-1, -1)

    def _parse_dir(self):

        cdir = os.path.join(self._dir, "config")
        self._machine = machine(os.path.join(cdir, 'machine.yaml'))

        self._exe = os.path.join(cdir, 'grace')

        self._info_file = os.path.join(cdir, "status.yaml")
        with open(self._info_file, "r") as f:
            self._lastjob = yaml.safe_load(f)

        pname = self._lastjob.get("parfile")
        if pname is not None:
            self._pfile = os.path.join(cdir, 'parfile', pname)
        else:
            # Fallback for simulations created before parfile was tracked in status.yaml
            parfile_dir = os.path.join(cdir, 'parfile')
            candidates = [f for f in os.listdir(parfile_dir) if os.path.isfile(os.path.join(parfile_dir, f))]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Could not determine parameter file: found {len(candidates)} files in {parfile_dir}. "
                    "Add a 'parfile' entry to status.yaml to disambiguate."
                )
            self._pfile = os.path.join(parfile_dir, candidates[0])

        if not os.path.isfile(self._pfile):
            raise RuntimeError(f"Parameter file not found: {self._pfile}")

        self._subscript = os.path.join(cdir, 'submission', "submission_script.x")

        if not os.path.isfile(self._subscript):
            raise RuntimeError("Could not find submission script")

        env_dir = os.path.join(cdir, 'env')
        if os.path.isdir(env_dir):
            candidates = [f for f in os.listdir(env_dir) if os.path.isfile(os.path.join(env_dir, f))]
            if len(candidates) == 1:
                self._env = os.path.join(env_dir, candidates[0])
            elif len(candidates) > 1:
                raise RuntimeError(
                    f"Multiple environment files found in {env_dir}. "
                    "Remove extras to disambiguate."
                )
            else:
                self._env = None
        else:
            self._env = None


        