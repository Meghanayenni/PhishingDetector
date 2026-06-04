import os
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


DATASET_FILE = "phishing_emails.csv"
MODEL_FILE = "phishing_email_model.pkl"


def create_sample_dataset():
    data = {
        "email": [
            "Your account has been suspended. Click here to verify your password immediately.",
            "Congratulations! You have won a free iPhone. Claim your prize now.",
            "Urgent! Your bank account will be blocked. Login using the link below.",
            "Verify your account now or your service will be terminated.",
            "You have received a secure message. Click here to view it.",
            "Your PayPal account has unusual activity. Confirm your identity now.",
            "Limited time offer! Claim your free reward by entering your card details.",
            "Your password expired. Update it immediately using this link.",
            "Security alert: Login detected from unknown device. Verify now.",
            "Your account needs verification. Failure to verify will block access.",
            "Dear user, click the attached link to restore your email access.",
            "Important bank notice: Confirm your PIN and account number.",
            "Your package delivery failed. Pay small fee to reschedule.",
            "Netflix billing failed. Update payment details to avoid suspension.",
            "You are selected for a cash reward. Submit your information now.",
            "Your cloud storage is full. Login here to keep your files.",
            "ATM card blocked. Reactivate by entering OTP and password.",
            "Final warning: Your account will be deleted today.",
            "Unusual login attempt. Click here to secure your account.",
            "You won a lottery. Send your details to receive the amount.",

            "Meeting is scheduled for tomorrow at 10 AM.",
            "Please find attached the project report for review.",
            "Your order has been shipped and will arrive tomorrow.",
            "Can we discuss the assignment after class?",
            "The college circular has been updated on the portal.",
            "Reminder: Submit your lab record before Friday.",
            "Your library book is due next week.",
            "The webinar link will be shared before the session.",
            "Please review the meeting minutes attached below.",
            "Your electricity bill has been generated successfully.",
            "Team lunch is planned for Saturday afternoon.",
            "The exam timetable has been released by the department.",
            "Your appointment with the doctor is confirmed.",
            "Thank you for attending the workshop.",
            "Please complete the feedback form by evening.",
            "Your train ticket booking is confirmed.",
            "The software update was installed successfully.",
            "Your monthly bank statement is available for download.",
            "Class has been rescheduled to 2 PM.",
            "Your parcel has been delivered successfully."
        ],
        "label": [
            "phishing", "phishing", "phishing", "phishing", "phishing",
            "phishing", "phishing", "phishing", "phishing", "phishing",
            "phishing", "phishing", "phishing", "phishing", "phishing",
            "phishing", "phishing", "phishing", "phishing", "phishing",

            "safe", "safe", "safe", "safe", "safe",
            "safe", "safe", "safe", "safe", "safe",
            "safe", "safe", "safe", "safe", "safe",
            "safe", "safe", "safe", "safe", "safe"
        ]
    }

    df = pd.DataFrame(data)
    df.to_csv(DATASET_FILE, index=False)
    print("Sample dataset created: phishing_emails.csv")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_dataset():
    if not os.path.exists(DATASET_FILE):
        create_sample_dataset()

    df = pd.read_csv(DATASET_FILE)

    if "email" not in df.columns or "label" not in df.columns:
        print("Dataset must contain two columns: email and label")
        exit()

    df = df.dropna()

    df["email"] = df["email"].apply(clean_text)
    df["label"] = df["label"].str.lower().str.strip()

    label_map = {
        "phishing": 1,
        "spam": 1,
        "malicious": 1,
        "safe": 0,
        "legitimate": 0,
        "ham": 0
    }

    df["label"] = df["label"].map(label_map)
    df = df.dropna()
    df["label"] = df["label"].astype(int)

    return df


def train_model():
    df = load_dataset()

    X = df["email"]
    y = df["label"]

    if len(df) < 4:
        print("Dataset is too small. Add more phishing and safe emails.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english"
        )),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\nMODEL TRAINING COMPLETED")
    print("------------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Safe", "Phishing"],
        zero_division=0
    ))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Safe", "Phishing"]
    )

    display.plot(cmap="Blues")
    plt.title("Confusion Matrix - Phishing Email Detection")
    plt.savefig("confusion_matrix.png")
    plt.show()

    joblib.dump(model, MODEL_FILE)

    print(f"\nModel saved as: {MODEL_FILE}")
    print("Confusion matrix saved as: confusion_matrix.png")


def predict_email():
    if not os.path.exists(MODEL_FILE):
        print("Model not found. Please train the model first.")
        return

    model = joblib.load(MODEL_FILE)

    print("\nPHISHING EMAIL DETECTION")
    print("------------------------")
    email = input("Enter email content: ")

    cleaned_email = clean_text(email)

    prediction = model.predict([cleaned_email])[0]
    probability = model.predict_proba([cleaned_email])[0]

    safe_probability = probability[0] * 100
    phishing_probability = probability[1] * 100

    print("\nPrediction Result:")

    if prediction == 1:
        print("This email is likely PHISHING.")
    else:
        print("This email is likely SAFE.")

    print(f"Safe Probability: {safe_probability:.2f}%")
    print(f"Phishing Probability: {phishing_probability:.2f}%")


def main():
    while True:
        print("\n==============================")
        print(" Phishing Email Detection Model")
        print("==============================")
        print("1. Train Model")
        print("2. Test Email")
        print("3. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            train_model()
        elif choice == "2":
            predict_email()
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()