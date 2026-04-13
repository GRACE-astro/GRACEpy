import argparse
import sys

import grace_pilot.simpilot as gp

def main():
    parser = argparse.ArgumentParser(
        description="SIMPilot CLI for creating and submitting simulations"
    )

    # Subparsers: first argument decides which command
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- create subcommand ----
    create_parser = subparsers.add_parser(
        "create", help="Create a new simulation"
    )
    create_parser.add_argument("--simname", required=True, help="Name of the simulation")
    create_parser.add_argument("--simpath", default=None, help="Path for the simulation")
    create_parser.add_argument("--machine", default=None, help="Machine to use")
    create_parser.add_argument("--executable", required=True, help="Path to executable")
    create_parser.add_argument("--parameter_file", required=True, help="Parameter file path")
    create_parser.add_argument("--env_file", default=None, help="Environment file (overrides machine default)")

    # ---- submit subcommand ----
    submit_parser = subparsers.add_parser(
        "submit", help="Submit a simulation"
    )
    submit_parser.add_argument("--simname", required=True, help="Name of the simulation to submit")
    submit_parser.add_argument("--queue", default=None, help="Queue to submit to")
    submit_parser.add_argument("--nodes", type=int, default=None)
    submit_parser.add_argument("--gpus_per_node", type=int, default=None)
    submit_parser.add_argument("--tasks_per_node", type=int, default=None)
    submit_parser.add_argument("--cpus_per_task", type=int, default=None)
    submit_parser.add_argument("--mem", type=int, default=None)
    submit_parser.add_argument("--mail_when", default=None)
    submit_parser.add_argument("--user_mail", default=None)
    submit_parser.add_argument("--walltime", default=None)

    # Parse arguments
    args = parser.parse_args()

    # Create an instance of simpilot
    sp = gp.simpilot()

    # Dispatch command
    if args.command == "create":
        sp.create_new_simulation(
            simname=args.simname,
            simpath=args.simpath,
            _machine=args.machine,
            executable=args.executable,
            parameter_file=args.parameter_file,
            env_file=args.env_file
        )
    elif args.command == "submit":
        sp.submit_simulation(
            simname=args.simname,
            queue=args.queue,
            nodes=args.nodes,
            gpus_per_node=args.gpus_per_node,
            tasks_per_node=args.tasks_per_node,
            cpus_per_task=args.cpus_per_task,
            mem=args.mem,
            mail_when=args.mail_when,
            user_mail=args.user_mail,
            walltime=args.walltime
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
