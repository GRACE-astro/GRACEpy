import os 
import yaml
import shutil 

from .machine import machine 
from .simulation import simulation 
from pathlib import Path


ALLOWED_MAIL_TYPE = ("NONE", "BEGIN", "END", "FAIL", "REQUEUE", "ALL", "INVALID_DEPEND", "STAGE_OUT", "TIME_LIMIT", "TIME_LIMIT_90", "TIME_LIMIT_80", "TIME_LIMIT_50", "ARRAY_TASKS")


class simpilot:
    
    def __init__(self):
        
        self._bdir = os.path.join(self.__get_path_raise_error("SIMPILOT_BASEDIR"), ".simpilot")

        self._grace_bdir = self.__get_path_raise_error("GRACE_HOME")

        self._detect_configuration()


    def __get_path_raise_error(self,name):
        path = os.environ[name]
        if path is None:
            raise RuntimeError(f"Environment variable {name} is not set.")
        return path 
    
    def _detect_configuration(self):
        
        need_setup = False 

        exists = os.path.isdir(self._bdir)

        # If readable but does not exist, we 
        # enter setup 
        if (not exists):
            os.makedirs(self._bdir)
            need_setup = True 
            

        # If the basedir is not readable, we're done 
        accessible = os.access(self._bdir, os.R_OK | os.X_OK)
        if not accessible:
            raise RuntimeError(f"Base directory {self._bdir} is not accessible.")

        
        
        # If exists but does not contain a full config,
        # we enter setup
        needed_files = ["known_machines.yaml", "user_settings.yaml"]
        config_detected = True 
        for ff in needed_files:
            if not os.path.isfile(os.path.join(self._bdir,ff)):
                config_detected=False 

        if not config_detected:
            need_setup = True 
        
        self._simrepo = os.path.join(self._bdir,"active_simulations")

        if need_setup:
            # Setup 
            self._setup_new_user()
        # If configuration is there, we parse it 
        self._parse_config() 

    def create_new_simulation(self,simname, simpath, machine_name, executable, parameter_file):

        if simpath is None:
            simpath = os.path.join(self._user_settings["simpath"], simname)
        if machine_name is None:
            _machine = machine(self._default_machine)
        else:
            _machine = machine(machine_name)
        if executable is None:
            raise ValueError("Invalid executable specified when creating a simulation")
        if parameter_file is None or (not os.path.isfile(parameter_file)):
            raise ValueError("Invalid parameter file specified when creating a simulation")
        
        submitscript = os.path.join(self._bdir,"submitscripts",_machine.submit_template)
        env_file = os.path.join(self._bdir,"env_files",_machine.env_file)

        sim = simulation(
            simname,simpath,_machine,
            env_file,submitscript,
            executable, parameter_file
        )
        # Write a descriptor of this simulation
        sim_descriptor = {
            "name": simname,
            "dir": simpath,
            "restart_id": -1
        }
        with open(os.path.join(self._simrepo,simname+".yaml"), "w") as f:
            yaml.safe_dump(sim_descriptor, f)


    def submit_simulation(self,simname,queue,nodes,gpus_per_node,tasks_per_node,cpus_per_task,mem,mail_when,user_mail,walltime):

        if not simname in self._active_sims:
            raise ValueError(f"Unknown simulation {simname}, create it first!")
        
        args = {}
        args["QUEUE"] = queue 
        args["GPUS_PER_NODE"] = gpus_per_node 
        args["N_NODES"] = nodes 
        args["TASKS_PER_NODE"] = tasks_per_node 
        args["CPUS_PER_TASK"] = cpus_per_task
        args["MEM"] = mem 
        args["MAIL_WHEN"] = mail_when if mail_when is not None else self._user_settings["mail_when"]
        args["USER_MAIL"] = user_mail if user_mail is not None else self._user_settings["email_address"]
        args["WALLTIME"] = walltime 

        # note these are slurm-like, for PBS we need a converter downstream of this (machine?)
        if args["MAIL_WHEN"] not in ALLOWED_MAIL_TYPE:
            raise ValueError("Invalid MAIL_WHEN argument")

        simfile = os.path.join(self._simrepo, simname+".yaml")
        with open(simfile,"r") as f: 
            simdata = yaml.safe_load(f)
        
        sim = simulation(
            simname, simdata["dir"]
        )

        sim.submit(args)

        

    def _parse_config(self):
        machines_file= os.path.join(self._bdir,"known_machines.yaml")
        with open(machines_file,"r") as f:
            machines = yaml.safe_load(f)
        self._known_machines = machines["known"]
        self._default_machine = os.path.join(self._bdir,"machines",machines["default"]+".yaml")

        settings_file =  os.path.join(self._bdir,"user_settings.yaml")
        with open(settings_file,"r") as f:
            self._user_settings = yaml.safe_load(f)

        self._active_sims = [f.stem for f in Path(self._simrepo).glob("*.yaml") if f.is_file()]

    def _setup_new_user(self):
        print("Setup required for simpilot.") 
        os.makedirs(self._simrepo, exist_ok=True)

        repo_path = input("simpilot needs a list of known machines, please provide the path of the GRACEpy repository: ")

        mfiles_path = os.path.join(repo_path,"known_machines")
        if not os.path.isdir(mfiles_path):
            raise ValueError("Invalid path.")
        
        known_machines = [f.stem for f in Path(mfiles_path).glob("*.yaml") if f.is_file()]

        machines_dir = os.path.join(self._bdir, "machines")
        os.makedirs(machines_dir,exist_ok=True)

        for m in known_machines:
            srcfile = os.path.join(mfiles_path,m+".yaml")
            dstfile = os.path.join(machines_dir,m+".yaml")
            shutil.copy(srcfile,dstfile)

        print(f"Auto-detected machines: {known_machines}")

        machines_info = {
            "known": known_machines
        }

        default_machine = input("Which machine is the default one on this system? ")
        if default_machine not in known_machines:
            raise ValueError(f"User-provided default machine {default_machine} not known.")

        machines_info["default"] = default_machine
        with open(os.path.join(self._bdir,"known_machines.yaml"),"w") as f:
            yaml.safe_dump(machines_info,f)

        user_defaults = {}

        email = input("Please provide your email: ")
        user_defaults["email_address"] = email 

        
        email_when = input(f"Please enter a default for the email-type of your job scripts \n(valid options: {ALLOWED_MAIL_TYPE}, default NONE): ")
        if email_when is None: email_when = "NONE"
        if email_when not in ALLOWED_MAIL_TYPE:
            raise  ValueError(f"User-provided default mail-when {email_when} is invalid.")
        
        user_defaults["mail_when"] = email_when

        default_simpath = input("Please enter a default path for simulations: ")
        user_defaults["simpath"] = default_simpath


        with open(os.path.join(self._bdir,"user_settings.yaml"),"w") as f:
            yaml.safe_dump(user_defaults,f)

        subpath = os.path.join(self._bdir, "submitscripts")
        os.makedirs(subpath, exist_ok=True)
        sfiles_path = os.path.join(mfiles_path, "submitscripts")
        for f in Path(sfiles_path).glob("*.sub"):
            if f.is_file():
                dstfile = os.path.join(subpath, f.name)
                shutil.copy(f, dstfile)

        subpath = os.path.join(self._bdir, "env_files")
        os.makedirs(subpath, exist_ok=True)
        sfiles_path = os.path.join(mfiles_path, "env_files")
        for f in Path(sfiles_path).glob("*.sh"):
            if f.is_file():
                dstfile = os.path.join(subpath, f.name)
                shutil.copy(f, dstfile)


        print("Done with initial configuration.")

        

