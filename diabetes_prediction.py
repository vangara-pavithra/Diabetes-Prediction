!pip install python-telegram-bot==13.7 --force-reinstall

from google.colab import drive
drive.mount('/content/drive')

!pip install transformers


import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the diabetes dataset
diabetes_data = pd.read_csv('/content/diabetes.csv')

# Split the dataset into features (X) and target (y)
X = diabetes_data.drop('Outcome', axis=1)
y = diabetes_data['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Define bot token and create bot instance
TOKEN = '6080390963:AAH6X1PG8Hyz0CDk99AoaAHW-zJAHDhr80M'
bot = telegram.Bot(token=TOKEN)

# Define the start command handler
def start(update, context):
    welcome_message = "Welcome to the Diabetes Prediction Bot!\n\n"
    welcome_message += "This bot can predict whether a person has diabetes or not based on their input values.\n"
    welcome_message += "Give the input values \nPregnancies \nGlucose \nBloodPressure \nSkinThickness \nInsulin \nBMI \nDiabetesPedigreeFunction \nAge \n \n"
    welcome_message += "To make a prediction, use the /predict command followed by eight numeric values separated by spaces.\n"
    welcome_message += "For example:\n"
    welcome_message += "/predict 5 166 72 19 175 25.8 0.587 51\n\n"
    welcome_message += "Please make sure to provide exactly eight numeric values."

    context.bot.send_message(chat_id=update.effective_chat.id, text=welcome_message)
# Define the prediction command handler
def predict_diabetes(update, context):
    message = update.message.text
    values = message.split(' ')[1:]
    if len(values) != 8:
        context.bot.send_message(chat_id=update.effective_chat.id, text="Invalid number of values provided.")
        return
    try:
        values = list(map(float, values))
    except ValueError:
        context.bot.send_message(chat_id=update.effective_chat.id, text="Invalid values provided.")
        return

    # Perform the diabetes prediction
    prediction = classifier.predict([values])

    result = "Diabetes" if prediction[0] == 1 else "No Diabetes"
    context.bot.send_message(chat_id=update.effective_chat.id, text=f"Prediction: {result}")

# Define the unknown command handler
def unknown(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I didn't understand that command.")

def main():
    updater = Updater(token=TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    # Add command handlers
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("predict", predict_diabetes))
    dispatcher.add_handler(MessageHandler(Filters.command, unknown))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
