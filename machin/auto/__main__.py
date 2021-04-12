import json
import argparse
from pprint import pprint
from machin.auto.config import (
    get_available_algorithms,
    get_available_environments,
    generate_algorithm_config,
    generate_env_config,
    generate_training_config,
    launch,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    p_list = subparsers.add_parser(
        "list", help="List available algorithms or environments."
    )

    p_list.add_argument(
        "--algo", action="store_true", help="List available algorithms.",
    )

    p_list.add_argument(
        "--env", action="store_true", help="List available environments."
    )

    p_generate = subparsers.add_parser("generate", help="Generate configuration.")

    p_generate.add_argument(
        "--algo", type=str, required=True, help="Algorithm name to use."
    )
    p_generate.add_argument(
        "--env", type=str, required=True, help="Environment name to use."
    )
    p_generate.add_argument(
        "--print", action="store_true", help="Direct config output to screen."
    )
    p_generate.add_argument(
        "--output",
        type=str,
        default="config.json",
        help="JSON config file output path.",
    )

    p_launch = subparsers.add_parser(
        "launch", help="Launch training with pytorch-lightning."
    )

    p_launch.add_argument(
        "--config", type=str, default="config.json", help="JSON config file path.",
    )

    args = parser.parse_args()
    if args.command == "list":
        if args.env:
            print("Available environments are:")
            for env in get_available_environments():
                print(env)
        elif args.algo:
            print("Available algorithms are:")
            for algo in get_available_algorithms():
                print(algo)
        else:
            print("You can list --algo or --env.")

    elif args.command == "generate":
        if args.algo not in get_available_algorithms():
            print(
                f"{args.algo} is not a valid algorithm, use list "
                "--algo to get a list of available algorithms."
            )
            exit(0)
        if args.env not in get_available_environments():
            print(
                f"{args.env} is not a valid environment, use list "
                "--env to get a list of available environments."
            )
            exit(0)
        config = {}
        config = generate_env_config(args.env, config=config)
        config = generate_algorithm_config(args.algo, config=config)
        config = generate_training_config(config=config)

        if args.print:
            pprint(config)

        with open(args.output, "w") as f:
            json.dump(config, f, indent=4, sort_keys=True)
        print(f"Config saved to {args.output}")

    elif args.command == "launch":
        with open(args.config, "r") as f:
            conf = json.load(f)
        launch(conf)
