import os 
import yaml
from .schedulers import slurm_scheduler

KNOWN_SCHEDULERS = ("SLURM", "PBS")

class machine:

    def __init__(self, fname):

        with open(fname,"r") as f:
            config = yaml.safe_load(f)
        
        self.name = config["name"]

        # What is the scheduler 
        self.scheduler_name = config["scheduler"]
        if not self.scheduler_name in KNOWN_SCHEDULERS:
            raise RuntimeError(f"Scheduler {self.scheduler_name} is not supported")
        if self.scheduler_name == "SLURM":
            self.scheduler = slurm_scheduler() 
        elif self.scheduler_name == "PBS":
            raise RuntimeError("PBS not implemented yet")

        # Max walltime in the queue
        self.max_walltime = config["max_walltime"]

        # What is the backend 
        self.backend = config["backend"]

        # Some hardware specs 
        self.mem_per_node = config["mem_per_node"]
        self.cpu_per_node = config["cpu_per_node"]
        self.gpu_per_node = config["gpu_per_node"]

        self.submit_template = config["submission_script"]
        self.env_file = config["environment_file"]
        self.queues = config["queues"]
        self.default_queue = config["default_queue"]

        self._config = config 

    def dump_config(self,path):
        with open(path,"w") as f:
            yaml.safe_dump(self._config,f)
    
    def check_submit_arguments_and_set_defaults(self, args):
        """Check submit arguments and fill in defaults if missing, raising errors for invalid values."""

        # Queue
        queue = args.get("QUEUE")
        if queue is None:
            args["QUEUE"] = self.default_queue
        elif queue not in self.queues:
            raise ValueError(f"Requested queue '{queue}' is not recognized")

        # GPUs per node
        gpus = args.get("GPUS_PER_NODE")
        if gpus is None:
            args["GPUS_PER_NODE"] = self.gpu_per_node
        elif not (1 <= gpus <= self.gpu_per_node):
            raise ValueError(f"Requested number of GPUs per node ({gpus}) is invalid, maximum is {self.gpu_per_node}")

        # Tasks per node
        tasks = args.get("TASKS_PER_NODE")
        if tasks is None:
            args["TASKS_PER_NODE"] = self.gpu_per_node
        elif not (1 <= tasks <= self.gpu_per_node):
            raise ValueError(f"Requested number of tasks per node ({tasks}) is invalid, maximum is {self.gpu_per_node}")

        # Memory per node
        mem = args.get("MEM")
        if mem is None:
            args["MEM"] = self.mem_per_node
        elif not (0 < mem <= self.mem_per_node):
            raise ValueError(f"Requested memory per node ({mem}) is invalid, maximum is {self.mem_per_node}")

        # CPUs per task
        cpus = args.get("CPUS_PER_TASK")
        if cpus is None:
            args["CPUS_PER_TASK"] = self.cpu_per_node // args["TASKS_PER_NODE"]
        elif not (0 < cpus <= self.cpu_per_node // args["TASKS_PER_NODE"]):
            raise ValueError(f"Requested CPUs per task ({cpus}) is invalid for {args['TASKS_PER_NODE']} tasks per node "
                            f"(maximum {self.cpu_per_node // args['TASKS_PER_NODE']})")

        



    def __str__(self):
        return (
            f"Machine: {self.name}\n"
            f"  Scheduler: {self.scheduler_name}\n"
            f"  Max walltime: {self.max_walltime}\n"
            f"  Backend: {self.backend}\n"
            f"  CPUs per node: {self.cpu_per_node}\n"
            f"  GPUs per node: {self.gpu_per_node}\n"
            f"  Memory per node: {self.mem_per_node} GB"
        )

    __repr__ = __str__
        

