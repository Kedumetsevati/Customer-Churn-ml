import numpy as np
import pandas as pd
from pathlib import Path
rng = np.random.default_rng(42)

N = 5000
# Categorical features
genders = rng.choice(["Male","Female"], size=N)
contract = rng.choice(["Month-to-month","One year","Two year"], p=[0.62,0.22,0.16], size=N)
internet = rng.choice(["DSL","Fiber optic","No"], p=[0.35,0.55,0.10], size=N)
payment = rng.choice(["Electronic check","Mailed check","Bank transfer","Credit card"],
                     p=[0.45,0.15,0.20,0.20], size=N)
phone = rng.choice(["Yes","No"], p=[0.85,0.15], size=N)
paperless = np.where(payment=="Electronic check","Yes", rng.choice(["Yes","No"], size=N))

# Numeric features
tenure = rng.integers(0, 72, size=N)
monthly = np.round(rng.normal(75, 30, size=N).clip(20, 200), 2)
dependents = rng.integers(0,5,size=N)
tech_support = rng.choice(["Yes","No"], p=[0.5,0.5], size=N)

# Create a latent churn probability (nonlinear + business logic)
logit = (
    -1.2
    + 0.03*(75-monthly)            # higher bill -> more churn
    - 0.02*tenure                  # longer tenure -> less churn
    + 0.6*(contract=="Month-to-month")
    + 0.4*(internet=="Fiber optic")
    + 0.25*(payment=="Electronic check")
    + 0.35*(paperless=="Yes")
    - 0.3*(tech_support=="Yes")
)

prob = 1/(1+np.exp(-logit))
churn = rng.binomial(1, prob)

df = pd.DataFrame({
    "customerID": [f"C{100000+i}" for i in range(N)],
    "gender": genders,
    "SeniorCitizen": rng.integers(0,2,size=N),
    "Partner": rng.choice(["Yes","No"], size=N),
    "DependentsCount": dependents,
    "PhoneService": phone,
    "MultipleLines": rng.choice(["Yes","No"], size=N),
    "InternetService": internet,
    "OnlineSecurity": rng.choice(["Yes","No"], size=N),
    "OnlineBackup": rng.choice(["Yes","No"], size=N),
    "DeviceProtection": rng.choice(["Yes","No"], size=N),
    "TechSupport": tech_support,
    "StreamingTV": rng.choice(["Yes","No"], size=N),
    "StreamingMovies": rng.choice(["Yes","No"], size=N),
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": np.round(monthly*np.maximum(tenure,1) + rng.normal(0, 20, N), 2),
    "Churn": np.where(churn==1, "Yes", "No")
})

# Save train data
out = Path("data") / "sample_churn.csv"
out.parent.mkdir(exist_ok=True, parents=True)
df.to_csv(out, index=False)

# small scoring sample
scoring = df.sample(20, random_state=1).drop(columns=["Churn"])
scoring.to_csv(Path("data")/"scoring_sample.csv", index=False)

print(f"Wrote {out} with shape {df.shape} and class balance: "
      f"{(df['Churn']=='Yes').mean():.2%} churn")
