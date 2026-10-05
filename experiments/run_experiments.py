import subprocess
import sys


def run_experiment(script):

    print("\n================================")
    print(f"Running: {script}")
    print("================================")

    result = subprocess.run(
        [
            sys.executable,
            script
        ]
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Experiment failed: {script}"
        )


def main():

    experiments = [
        "experiments/compare_newton.py",
        "experiments/scaling_experiment.py"
    ]

    for experiment in experiments:

        run_experiment(
            experiment
        )

    print("\n================================")
    print("All experiments completed.")
    print("================================")


if __name__ == "__main__":
    main()