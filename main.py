import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/Myths_and_Facts_Data_on_Child_Development.csv")

print("Original dataset shape:", df.shape)


# --------------------------------------------------
# 2. Basic data cleaning
# --------------------------------------------------

# Remove exact duplicate rows
df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())


# --------------------------------------------------
# 3. Create Type and Strength columns
# --------------------------------------------------

df["Type"] = df["Label"].str.extract(
    r"(facts|myths)",
    expand=False
).str.title()

df["Strength"] = df["Label"].str.extract(
    r"(Strong|Moderate|Weak|Hardly)",
    expand=False
)


# --------------------------------------------------
# 4. Create text-based features
# --------------------------------------------------

df["statement_length"] = df["Statements"].str.len()

df["word_count"] = (
    df["Statements"]
    .str.split()
    .str.len()
)


# --------------------------------------------------
# 5. Basic statistics
# --------------------------------------------------

print("\nType counts:")
print(df["Type"].value_counts())

print("\nStrength counts:")
print(df["Strength"].value_counts())

print("\nPercentage of each type:")
print(
    (df["Type"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nAverage word count:")
print(
    df.groupby("Type")["word_count"]
    .mean()
    .round(2)
)

print("\nMedian word count:")
print(
    df.groupby("Type")["word_count"]
    .median()
)


# --------------------------------------------------
# 6. Strength analysis
# --------------------------------------------------

strength_type = pd.crosstab(
    df["Strength"],
    df["Type"]
)

print("\nFacts vs Myths by strength:")
print(strength_type)


strength_percentage = (
    pd.crosstab(
        df["Type"],
        df["Strength"],
        normalize="index"
    ) * 100
)

print("\nStrength percentage within each type:")
print(
    strength_percentage.round(2)
)


myth_rate_by_strength = (
    pd.crosstab(
        df["Strength"],
        df["Type"],
        normalize="index"
    ) * 100
)

print("\nMyth percentage within each strength:")
print(
    myth_rate_by_strength.round(2)
)


# --------------------------------------------------
# 7. Create visuals
# --------------------------------------------------

# Facts vs Myths

df["Type"].value_counts().plot(kind="bar")

plt.title("Facts vs Myths")
plt.xlabel("Type")
plt.ylabel("Number of Statements")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "visuals/facts_vs_myths.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Facts vs Myths by strength

strength_type.plot(kind="bar")

plt.title("Facts vs Myths by Strength")
plt.xlabel("Strength")
plt.ylabel("Number of Statements")
plt.xticks(rotation=0)
plt.legend(title="Type")
plt.tight_layout()

plt.savefig(
    "visuals/facts_vs_myths_by_strength.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Average word count

average_words = (
    df.groupby("Type")["word_count"]
    .mean()
)

average_words.plot(kind="bar")

plt.title("Average Word Count: Facts vs Myths")
plt.xlabel("Type")
plt.ylabel("Average Number of Words")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "visuals/average_word_count.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Myth percentage by strength

myth_percentage = (
    myth_rate_by_strength["Myths"]
)

myth_percentage.plot(kind="bar")

plt.title("Percentage of Myths by Strength")
plt.xlabel("Strength")
plt.ylabel("Percentage of Statements")
plt.xticks(rotation=0)
plt.ylim(0, 100)
plt.tight_layout()

plt.savefig(
    "visuals/myth_percentage_by_strength.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 8. Save cleaned dataset
# --------------------------------------------------

df.to_csv(
    "data/childhood_myths_cleaned.csv",
    index=False
)

print("\nProject analysis completed successfully!")