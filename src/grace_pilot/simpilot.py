import os
import yaml
import shutil

from .machine import machine
from .simulation import simulation
from pathlib import Path


ALLOWED_MAIL_TYPE = ("NONE", "BEGIN", "END", "FAIL", "REQUEUE", "ALL", "INVALID_DEPEND", "STAGE_OUT", "TIME_LIMIT", "TIME_LIMIT_90", "TIME_LIMIT_80", "TIME_LIMIT_50", "ARRAY_TASKS")

# External initial-data libraries simpilot asks about at setup. Their install
# dirs are written into the per-user site file (site.sh), which every
# environment file sources. HOME_KADATH in particular is read by FUKA via
# getenv() at RUNTIME, so it must be present in the job environment or FUKA
# initial-data reads segfault.
EXTERNAL_LIBS = (
    ("FUKA / Kadath", "HOME_KADATH"),
    ("LORENE",        "HOME_LORENE"),
    ("TwoPunctures",  "TwoPunctures_ROOT"),
)
SITE_FILE_PLACEHOLDER = "@SITE_FILE@"


class simpilot:

    def __init__(self):

        self._bdir = os.path.join(self._get_env_path("SIMPILOT_BASEDIR"), ".simpilot")

        self._grace_bdir = self._get_env_path("GRACE_HOME")

        self._detect_configuration()


    def _get_env_path(self, name):
        path = os.environ.get(name)
        if path is None:
            raise RuntimeError(f"Environment variable {name} is not set.")
        return path

    def _detect_configuration(self):

        need_setup = False

        if not os.path.isdir(self._bdir):
            os.makedirs(self._bdir)
            need_setup = True
        elif not os.access(self._bdir, os.R_OK | os.X_OK):
            raise RuntimeError(f"Base directory {self._bdir} is not accessible.")

        # If exists but does not contain a full config,
        # we enter setup
        needed_files = ["known_machines.yaml", "user_settings.yaml"]
        for ff in needed_files:
            if not os.path.isfile(os.path.join(self._bdir, ff)):
                need_setup = True
                break

        self._simrepo = os.path.join(self._bdir, "active_simulations")

        if need_setup:
            self._setup_new_user()
        else:
            self._parse_config()

    def _resolve_machine(self, machine_name):
        machine_file = os.path.join(self._bdir, "machines", machine_name + ".yaml")
        if not os.path.isfile(machine_file):
            raise ValueError(f"No configuration file found for machine '{machine_name}'")
        return machine(machine_file)

    def create_new_simulation(self, simname, simpath=None, _machine=None, executable=None, parameter_file=None, env_file=None):

        if simpath is None:
            simpath = os.path.join(self._user_settings["simpath"], simname)
        if _machine is None:
            _machine = self._resolve_machine(self._default_machine)
        elif isinstance(_machine, str):
            _machine = self._resolve_machine(_machine)
        if executable is None:
            raise ValueError("Invalid executable specified when creating a simulation")
        if parameter_file is None or (not os.path.isfile(parameter_file)):
            raise ValueError("Invalid parameter file specified when creating a simulation")

        submitscript = os.path.join(self._bdir, "submitscripts", _machine.submit_template)
        if env_file is not None:
            envfile = env_file
        else:
            envfile = os.path.join(self._bdir, "env_files", _machine.env_file)

        sim = simulation(
            simname, simpath, _machine,
            submitscript,
            executable, parameter_file,
            env=envfile
        )
        # Write a descriptor of this simulation
        sim_descriptor = {
            "name": simname,
            "dir": simpath,
            "restart_id": -1
        }
        with open(os.path.join(self._simrepo, simname + ".yaml"), "w") as f:
            yaml.safe_dump(sim_descriptor, f)


    def submit_simulation(self, simname, queue=None, nodes=None, gpus_per_node=None,
                          tasks_per_node=None, cpus_per_task=None, mem=None,
                          mail_when=None, user_mail=None, walltime=None):

        if simname not in self._active_sims:
            raise ValueError(f"Unknown simulation {simname}, create it first!")

        if nodes is None:
            raise ValueError("Number of nodes must be specified")
        if walltime is None:
            raise ValueError("Walltime must be specified")

        args = {}
        args["QUEUE"] = queue
        args["GPUS_PER_NODE"] = gpus_per_node
        args["N_NODES"] = nodes
        args["TASKS_PER_NODE"] = tasks_per_node
        args["CPUS_PER_TASK"] = cpus_per_task
        args["MEM"] = mem
        args["MAIL_WHEN"] = mail_when if mail_when is not None else self._user_settings.get("mail_when", "NONE")
        args["USER_MAIL"] = user_mail if user_mail is not None else self._user_settings.get("email_address")
        args["WALLTIME"] = walltime

        # note these are slurm-like, for PBS we need a converter downstream of this (machine?)
        if args["MAIL_WHEN"] not in ALLOWED_MAIL_TYPE:
            raise ValueError(f"Invalid MAIL_WHEN argument: '{args['MAIL_WHEN']}'")

        simfile = os.path.join(self._simrepo, simname + ".yaml")
        with open(simfile, "r") as f:
            simdata = yaml.safe_load(f)

        sim = simulation(simname, simdata["dir"])

        sim.submit(args)

        

    def _parse_config(self):
        machines_file = os.path.join(self._bdir, "known_machines.yaml")
        with open(machines_file, "r") as f:
            machines = yaml.safe_load(f)
        self._known_machines = machines["known"]
        self._default_machine = machines["default"]

        settings_file = os.path.join(self._bdir, "user_settings.yaml")
        with open(settings_file, "r") as f:
            self._user_settings = yaml.safe_load(f)

        os.makedirs(self._simrepo, exist_ok=True)
        self._active_sims = [f.stem for f in Path(self._simrepo).glob("*.yaml") if f.is_file()]

    def _write_site_file(self, ext_libs):
        """Write the per-user site file sourced by every environment file.

        Exports the install dirs the user provided and leaves a commented
        placeholder for the ones left blank, so they can be filled in later.
        Returns the absolute path to the written file.
        """
        site_file = os.path.join(self._bdir, "site.sh")
        lines = [
            "#!/bin/sh",
            "# simpilot site file: install dirs for external initial-data libraries.",
            "# Sourced by every job's environment file. Edit to add or change paths.",
            "# HOME_KADATH is read by FUKA at runtime, so it must be set for FUKA runs.",
            "",
        ]
        for label, var in EXTERNAL_LIBS:
            val = ext_libs.get(var, "")
            if val:
                lines.append(f'export {var}="{val}"   # {label}')
            else:
                lines.append(f'# export {var}=   # {label}')
        with open(site_file, "w") as f:
            f.write("\n".join(lines) + "\n")
        return site_file

    def _setup_new_user(self):
        print("Setup required for simpilot.")
        os.makedirs(self._simrepo, exist_ok=True)

        repo_path = input("simpilot needs a list of known machines, please provide the path of the GRACEpy repository: ").strip()

        mfiles_path = os.path.join(repo_path, "known_machines")
        if not os.path.isdir(mfiles_path):
            raise ValueError(f"Invalid path: {mfiles_path} does not exist.")

        known_machines = [f.stem for f in Path(mfiles_path).glob("*.yaml") if f.is_file()]
        if not known_machines:
            raise ValueError(f"No machine configurations found in {mfiles_path}")

        machines_dir = os.path.join(self._bdir, "machines")
        os.makedirs(machines_dir, exist_ok=True)

        for m in known_machines:
            srcfile = os.path.join(mfiles_path, m + ".yaml")
            dstfile = os.path.join(machines_dir, m + ".yaml")
            shutil.copyfile(srcfile, dstfile)

        print(f"Auto-detected machines: {known_machines}")

        machines_info = {
            "known": known_machines
        }

        if len(known_machines) == 1:
            default_machine = known_machines[0]
            print(f"Only one machine found, using '{default_machine}' as default.")
        else:
            default_machine = input("Which machine is the default one on this system? ").strip()
        if default_machine not in known_machines:
            raise ValueError(f"User-provided default machine '{default_machine}' not known. Available: {known_machines}")

        machines_info["default"] = default_machine
        with open(os.path.join(self._bdir, "known_machines.yaml"), "w") as f:
            yaml.safe_dump(machines_info, f)

        user_defaults = {}

        email = input("Please provide your email: ").strip()
        user_defaults["email_address"] = email

        email_when = input(f"Please enter a default for the email-type of your job scripts\n(valid options: {ALLOWED_MAIL_TYPE}, default NONE): ").strip()
        if not email_when:
            email_when = "NONE"
        if email_when not in ALLOWED_MAIL_TYPE:
            raise ValueError(f"User-provided default mail-when '{email_when}' is invalid.")

        user_defaults["mail_when"] = email_when

        default_simpath = input("Please enter a default path for simulations: ").strip()
        user_defaults["simpath"] = default_simpath

        # Install dirs for external initial-data libraries. Optional: a blank
        # answer leaves a commented placeholder in the site file to edit later.
        print("\nInstall directories for external initial-data libraries (optional).")
        print("Leave blank to skip; you can edit them in later.")
        ext_libs = {}
        for label, var in EXTERNAL_LIBS:
            ext_libs[var] = input(f"  {label} [{var}]: ").strip()
        user_defaults["external_libs"] = ext_libs

        with open(os.path.join(self._bdir, "user_settings.yaml"), "w") as f:
            yaml.safe_dump(user_defaults, f)

        site_file = self._write_site_file(ext_libs)
        if any(ext_libs.values()):
            print(f"External library paths written to {site_file}")
        else:
            print(f"No external library paths set. Edit {site_file} to add them later.")

        # Copy submit script templates
        subpath = os.path.join(self._bdir, "submitscripts")
        os.makedirs(subpath, exist_ok=True)
        sfiles_path = os.path.join(mfiles_path, "submitscripts")
        if os.path.isdir(sfiles_path):
            for f in Path(sfiles_path).iterdir():
                if f.is_file():
                    shutil.copyfile(f, os.path.join(subpath, f.name))

        # Copy environment files, substituting the absolute path to the site
        # file so each job sources the user's external-library install dirs.
        envpath = os.path.join(self._bdir, "env_files")
        os.makedirs(envpath, exist_ok=True)
        efiles_path = os.path.join(mfiles_path, "env_files")
        if os.path.isdir(efiles_path):
            for f in Path(efiles_path).iterdir():
                if f.is_file():
                    content = f.read_text().replace(SITE_FILE_PLACEHOLDER, site_file)
                    Path(os.path.join(envpath, f.name)).write_text(content)

        print("Done with initial configuration.")
        self._parse_config()


