import pandas as pd

def load_and_preprocess():
    df = pd.read_csv("laptop_battery_health_usage.csv")

    df.columns = df.columns.str.strip().str.lower()

    if "device_id" in df.columns:
        df = df.drop("device_id", axis=1)


    df["overheating_issues"] = df["overheating_issues"].map({"Yes":1, "No":0})

    df["replace_battery"] = df["battery_health_percent"].apply(
        lambda x: 1 if x < 70 else 0
    )


    df = pd.get_dummies(df, columns=["brand","os","usage_type"], drop_first=True)

    return df