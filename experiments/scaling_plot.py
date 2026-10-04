import csv
import matplotlib.pyplot as plt


def load_results(filename):
    results = []

    with open(filename, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            results.append({
                "dimension": int(row["dimension"]),
                "shots": int(row["shots"]),
                "runtime": float(row["runtime_seconds"])
            })

    return results


def main():

    filename = "experiments/results/scaling_results.csv"

    results = load_results(filename)

    shots_values = sorted(
        set(row["shots"] for row in results)
    )

    plt.figure()

    for shots in shots_values:

        data = [
            row for row in results
            if row["shots"] == shots
        ]

        data.sort(key=lambda row: row["dimension"])

        dimensions = [
            row["dimension"] for row in data
        ]

        runtimes = [
            row["runtime"] for row in data
        ]

        plt.plot(
            dimensions,
            runtimes,
            marker="o",
            label=f"{shots} shots"
        )

    plt.xlabel("Problem Dimension")
    plt.ylabel("Runtime (seconds)")
    plt.title("QNM Scaling Experiment")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        "experiments/results/scaling_runtime.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("Plot saved to:")
    print("experiments/results/scaling_runtime.png")


if __name__ == "__main__":
    main()