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

    def __init__(self,name, dir,machine=None,submitscript=None,exe=None,parfile=None):
        self._name = name 
        self._dir = dir 
        self._cdir = os.path.join(self._dir, "config")
        self._machine = machine 
        self._exe = exe
        self._pfile = parfile 
        self._subscript = submitscript

        if not os.path.isdir(dir):
            self._init_directory_structure()
        else:
            self._parse_dir()

    
    def submit(self,sub_args):
        rid = int(self._lastjob["restart_id"]) + 1
        prev_jobid = int(self._lastjob["job_id"])

        sub_args["JOBNAME"] = self._name + f"-{rid:04d}"

        simdir = os.path.join(self._dir, f"restart_{rid:04d}")
        os.makedirs(simdir)

        _ = self._copyfile(self._exe,simdir)
        pfile = self._copyfile(self._pfile,simdir)
        sub_args["PARAMETER_FILE"] = pfile
        sfile = self._edit_submit_script(simdir,sub_args)

        # check whether chaining is necessary 
        need_chain = False 
        if (rid > 0):
            status = self._machine.scheduler.getstatus(prev_jobid)
            need_chain = status in CHAIN_STATES

        if need_chain:
            jobid = self._machine.scheduler.chain_submission(sfile,prev_jobid)
            print(f"Submitted simulation {self._name}, restart id {rid}, job id {jobid} with dependency {prev_jobid}")
        else:
            jobid = self._machine.scheduler.submit(sfile)
            print(f"Submitted simulation {self._name}, restart id {rid}, job id {jobid}")
        self._update_last_job(rid,jobid)
    


        
    def _update_last_job(self,rid,jid):
        self._lastjob["job_id"] = jid 
        self._lastjob["restart_id"] = rid 
        with open(self._info_file, "w") as f:
            yaml.safe_dump(self._lastjob,f)

    
    def _edit_submit_script(self,simdir,args):
        spath,sname = os.path.split(self._subscript)
        subfile = os.path.join(simdir,sname)
        self._machine.check_submit_arguments(args)
        fill_submit_template(self._subscript,subfile)
        return subfile 

    def _copyfile(self,srcfile,dstpath):
        pth,nm = os.path.split(srcfile)
        dstfile = os.path.join(dstpath,nm)
        shutil.copy(srcfile,dstfile)
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
        
        self._machine.dump_config(os.path.join(cdir,"machine.yaml"))
        
        shutil.copyfile(self._exe, os.path.join(cdir,'grace'))

        ppath,pname = os.path.split(self._pfile)
        os.makedirs(os.path.join(cdir,'parfile'))
        shutil.copyfile(self._pfile,os.path.join(cdir,'parfile',pname))

        spath,sname = os.path.split(self._subscript)
        os.makedirs(os.path.join(cdir,'submission'))
        shutil.copyfile(self._subscript,os.path.join(cdir,'submission',"submission_script.x"))

        
        self._info_file = os.path.join(cdir,"status.yaml")

        self._update_last_job(-1,-1)        

    def _parse_dir(self):

        cdir = os.path.join(self._dir,"config")
        self._machine = machine(os.path.join(cdir,'machine.yaml'))

        self._exe = os.path.join(cdir,'grace')

        parfile_dir = os.path.join(cdir,'parfile')
        root, _, files = next(os.walk(parfile_dir))

        if not len(files) == 1:
            raise RuntimeError("Could not determine parameter file")
        
        self._pfile = os.path.join(root,files[0])

        self._subscript = os.path.join(cdir,'submission',"submission_script.x")

        if not os.path.isfile(self._subscript):
            raise RuntimeError("Could not find submission script")

        self._info_file = os.path.join(cdir,"status.yaml")
        with open(self._info_file, "r") as f:
            self._lastjob = yaml.safe_load(f)




        


        