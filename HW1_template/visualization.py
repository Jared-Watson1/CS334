import matplotlib.pyplot as plt
import seaborn as sns
import palmer


# part a
def plot_histogram_before_removing_NaN():
    penguins_df = palmer.load_csv("penguins.csv")

    plt.figure(figsize=(10, 6))
    penguins_df["species"].value_counts().plot(kind="bar")
    plt.title("Species Count Before Removing NaN")
    plt.xlabel("Species")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()


def plot_histogram_after_removing_NaN():
    penguins_df = palmer.load_csv("penguins.csv")

    col_name = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]

    penguins_df_clean = penguins_df.dropna(subset=col_name)

    plt.figure(figsize=(10, 6))
    penguins_df_clean["species"].value_counts().plot(kind="bar")
    plt.title("Species Count After Removing NaN")
    plt.xlabel("Species")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

    return penguins_df_clean


# part b
def plot_bill_length_distribution():
    penguins_df = palmer.load_csv("penguins.csv")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="species", y="bill_length_mm", data=penguins_df)
    plt.title("Bill Length Distribution by Species")
    plt.xlabel("Species")
    plt.ylabel("Bill Length (mm)")
    plt.show()


def plot_bill_depth_distribution():
    penguins_df = palmer.load_csv("penguins.csv")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="species", y="bill_depth_mm", data=penguins_df)
    plt.title("Bill Depth Distribution by Species")
    plt.xlabel("Species")
    plt.ylabel("Bill Depth (mm)")
    plt.show()


def plot_flipper_length_distribution():
    penguins_df = palmer.load_csv("penguins.csv")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="species", y="flipper_length_mm", data=penguins_df)
    plt.title("Flipper Length Distribution by Species")
    plt.xlabel("Species")
    plt.ylabel("Flipper Length (mm)")
    plt.show()


def plot_body_mass_distribution():
    penguins_df = palmer.load_csv("penguins.csv")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="species", y="body_mass_g", data=penguins_df)
    plt.title("Body Mass Distribution by Species")
    plt.xlabel("Species")
    plt.ylabel("Body Mass (g)")
    plt.show()


# part c
def plot_bill_length_vs_bill_depth():
    penguins_df = palmer.load_csv("penguins.csv")
    sns.scatterplot(
        x="bill_length_mm", y="bill_depth_mm", hue="species", data=penguins_df
    )
    plt.title("Bill Length vs. Bill Depth by Species")
    plt.xlabel("Bill Length (mm)")
    plt.ylabel("Bill Depth (mm)")
    plt.legend(title="Species")
    plt.show()


def plot_bill_length_vs_flipper_length():
    penguins_df = palmer.load_csv("penguins.csv")
    sns.scatterplot(
        x="bill_length_mm", y="flipper_length_mm", hue="species", data=penguins_df
    )
    plt.title("Bill Length vs. Flipper Length by Species")
    plt.xlabel("Bill Length (mm)")
    plt.ylabel("Flipper Length (mm)")
    plt.legend(title="Species")
    plt.show()


def plot_bill_length_vs_body_mass():
    penguins_df = palmer.load_csv("penguins.csv")
    sns.scatterplot(
        x="bill_length_mm", y="body_mass_g", hue="species", data=penguins_df
    )
    plt.title("Bill Length vs. Body Mass by Species")
    plt.xlabel("Bill Length (mm)")
    plt.ylabel("Body Mass (g)")
    plt.legend(title="Species")
    plt.show()


def plot_bill_depth_vs_flipper_length():
    penguins_df = palmer.load_csv("penguins.csv")
    sns.scatterplot(
        x="bill_depth_mm", y="flipper_length_mm", hue="species", data=penguins_df
    )
    plt.title("Bill Depth vs. Flipper Length by Species")
    plt.xlabel("Bill Depth (mm)")
    plt.ylabel("Flipper Length (mm)")
    plt.legend(title="Species")
    plt.show()


def plot_bill_depth_vs_body_mass():
    penguins_df = palmer.load_csv("penguins.csv")
    sns.scatterplot(x="bill_depth_mm", y="body_mass_g", hue="species", data=penguins_df)
    plt.title("Bill Depth vs. Body Mass by Species")
    plt.xlabel("Bill Depth (mm)")
    plt.ylabel("Body Mass (g)")
    plt.legend(title="Species")
    plt.show()


def plot_flipper_length_vs_body_mass():
    penguins_df = palmer.load_csv("penguins.csv")
    sns.scatterplot(
        x="flipper_length_mm", y="body_mass_g", hue="species", data=penguins_df
    )
    plt.title("Flipper Length vs. Body Mass by Species")
    plt.xlabel("Flipper Length (mm)")
    plt.ylabel("Body Mass (g)")
    plt.legend(title="Species")
    plt.show()


def main():
    # plot_bill_length_vs_bill_depth()
    plot_bill_length_vs_flipper_length()
    # plot_bill_length_vs_body_mass()
    # plot_bill_depth_vs_flipper_length()
    # plot_bill_depth_vs_body_mass()
    # plot_flipper_length_vs_body_mass()


if __name__ == "__main__":
    main()
