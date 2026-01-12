from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, CallbackContext, MessageHandler, filters
import hashlib
import base64
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import requests
from solders.pubkey import Pubkey as PublicKey
from solana.rpc.api import Client

# Define your wallet addresses
BITCOIN_WALLET_ADDRESS = "bc1q26yxjmxh9l3ds9msccuqv85wuatjhq5qy2k5jv"
SOLANA_WALLET_ADDRESS = "9r9DmsYeHdYpSYBfgXLFiPUs17wcoCg7geKPocAKRPd"

# Define subscription plans
SUBSCRIPTION_PLANS = {
    "10_days": {"price_btc": 0.001, "price_sol": 0.1, "duration": timedelta(days=10)},
    "30_days": {"price_btc": 0.003, "price_sol": 0.3, "duration": timedelta(days=30)},
    "lifetime": {"price_btc": 0.01, "price_sol": 1.0, "duration": timedelta(days=365*10)},
}

ALLOWED_CHAT_IDS = [8254145022]

# Load and train your machine learning models
def load_and_train_models():
    models = {}
    try:
        # Example: Load historical data roulette
        roulette_data = pd.read_csv('roulette_historical_data.csv')
        X_roulette = roulette_data.drop('outcome', axis=1)
        y_roulette = roulette_data['outcome']
        X_train, X_test, y_train, y_test = train_test_split(X_roulette, y_roulette, test_size=0.2, random_state=42)
        model_roulette = RandomForestClassifier(n_estimators=100, random_state=42)
        model_roulette.fit(X_train, y_train)
        accuracy = metrics.accuracy_score(y_test, model_roulette.predict(X_test))
        print(f'Roulette Model Accuracy: {accuracy}')
        models['roulette'] = model_roulette
    except FileNotFoundError:
        print("Roulette historical data not found. Model not loaded.")

    try:
        # Load historical data blackjack
        blackjack_data = pd.read_csv('blackjack_historical_data.csv')
        X_blackjack = blackjack_data.drop('outcome', axis=1)
        y_blackjack = blackjack_data['outcome']
        X_train, X_test, y_train, y_test = train_test_split(X_blackjack, y_blackjack, test_size=0.2, random_state=42)
        model_blackjack = RandomForestClassifier(n_estimators=100, random_state=42)
        model_blackjack.fit(X_train, y_train)
        accuracy = metrics.accuracy_score(y_test, model_blackjack.predict(X_test))
        print(f'Blackjack Model Accuracy: {accuracy}')
        models['blackjack'] = model_blackjack
    except FileNotFoundError:
        print("Blackjack historical data not found. Model not loaded.")

    # Repeat for other games...
    # slots_data = pd.read_csv('slots_historical_data.csv')
    # model_slots = ...

    return models

MODELS = load_and_train_models()

# Define the start command
async def start(update: Update, context: CallbackContext) -> None:
    keyboard = [
        [InlineKeyboardButton("Roulette 🎰", callback_data='roulette')],
        [InlineKeyboardButton("Blackjack 🃏", callback_data='blackjack')],
        [InlineKeyboardButton("Slots 🎰", callback_data='slots')],
        [InlineKeyboardButton("Poker 🃏", callback_data='poker')],
        [InlineKeyboardButton("Baccarat 🃏", callback_data='baccarat')],
        [InlineKeyboardButton("Subscribe 💰", callback_data='subscribe')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('🎉 Welcome to Bot Oracle! 🔮 Make any Prediction with this bot! 💰 First Pay For a Plan to Get Access To Featuring!', reply_markup=reply_markup)

# Define the subscribe command
async def subscribe(update: Update, context: CallbackContext) -> None:
    text = f'To subscribe, please send a payment to the following addresses:\n\nBitcoin: {BITCOIN_WALLET_ADDRESS}\nSolana: {SOLANA_WALLET_ADDRESS}\n\nChoose a plan:\n10 Days: 0.001 BTC / 0.1 SOL\n30 Days: 0.003 BTC / 0.3 SOL\nLifetime: 0.01 BTC / 1.0 SOL\n\nOnce paid, type /verify to gain access.'
    keyboard = [[InlineKeyboardButton("⬅️ Back", callback_data='back')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(text, reply_markup=reply_markup)

# Define the button callback handler
async def button(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    await query.answer()

    data = query.data
    if data == 'subscribe':
        keyboard = [
            [InlineKeyboardButton("10 Days - 0.001 BTC/0.1 SOL 💰", callback_data='subscribe_10_days')],
            [InlineKeyboardButton("30 Days - 0.003 BTC/0.3 SOL 💰", callback_data='subscribe_30_days')],
            [InlineKeyboardButton("Lifetime - 0.01 BTC/1.0 SOL 💰", callback_data='subscribe_lifetime')],
            [InlineKeyboardButton("⬅️ Back", callback_data='back')],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text='Choose a subscription plan:', reply_markup=reply_markup)
    elif data.startswith('subscribe_'):
        plan = data.split('_')[1] + '_' + data.split('_')[2] if len(data.split('_')) > 2 else data.split('_')[1]
        price_btc = SUBSCRIPTION_PLANS[plan]['price_btc']
        price_sol = SUBSCRIPTION_PLANS[plan]['price_sol']
        keyboard = [[InlineKeyboardButton("⬅️ Back", callback_data='back')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text=f'To subscribe to the {plan.replace("_", " ")} plan, please send a payment of {price_btc} BTC or {price_sol} SOL to the following addresses:\n\nBitcoin: {BITCOIN_WALLET_ADDRESS}\nSolana: {SOLANA_WALLET_ADDRESS}\n\nOnce paid, type /verify to gain access.', reply_markup=reply_markup)
        context.user_data['subscription_plan'] = plan
    elif data == 'back':
        keyboard = [
            [InlineKeyboardButton("Roulette 🎰", callback_data='roulette')],
            [InlineKeyboardButton("Blackjack 🃏", callback_data='blackjack')],
            [InlineKeyboardButton("Slots 🎰", callback_data='slots')],
            [InlineKeyboardButton("Poker 🃏", callback_data='poker')],
            [InlineKeyboardButton("Baccarat 🃏", callback_data='baccarat')],
            [InlineKeyboardButton("Subscribe 💰", callback_data='subscribe')],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text='🎉 Welcome to Bot Oracle! 🔮 Make any Prediction with this bot! 💰 First Pay For a Plan to Get Access To Featuring!', reply_markup=reply_markup)
    elif data in ['roulette', 'blackjack', 'slots', 'poker', 'baccarat']:
        if query.message.chat.id in ALLOWED_CHAT_IDS or context.user_data.get('subscribed'):
            keyboard = [[InlineKeyboardButton("⬅️ Back", callback_data='back')]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(text=f'You selected {data}. Please provide the server seed, client seed, and nonce.', reply_markup=reply_markup)
            context.user_data['game'] = data
        else:
            keyboard = [[InlineKeyboardButton("⬅️ Back", callback_data='back')]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(text='Please subscribe to access this feature. Use the /subscribe command to learn more.', reply_markup=reply_markup)

# Define the seed input handler
async def seed_input(update: Update, context: CallbackContext) -> None:
    if update.effective_chat.id not in ALLOWED_CHAT_IDS and not context.user_data.get('subscribed'):
        keyboard = [[InlineKeyboardButton("⬅️ Back", callback_data='back')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text('Please subscribe to access this feature. Use the /subscribe command to learn more.', reply_markup=reply_markup)
        return

    user_input = update.message.text.split()
    if len(user_input) != 3:
        keyboard = [[InlineKeyboardButton("⬅️ Back", callback_data='back')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text('Please provide the server seed, client seed, and nonce separated by spaces.', reply_markup=reply_markup)
        return
    server_seed, client_seed, nonce = user_input
    game = context.user_data.get('game')

    if game in MODELS:
        prediction = predict_game(MODELS[game], server_seed, client_seed, nonce)
        keyboard = [[InlineKeyboardButton("⬅️ Back", callback_data='back')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(f'The predicted outcome for {game} is: {prediction}', reply_markup=reply_markup)
    else:
        keyboard = [[InlineKeyboardButton("⬅️ Back", callback_data='back')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text('Game not supported. Please choose a valid game.', reply_markup=reply_markup)

# Define the verification command
async def verify(update: Update, context: CallbackContext) -> None:
    plan = context.user_data.get('subscription_plan')
    if plan:
        duration = SUBSCRIPTION_PLANS[plan]['duration']
        context.user_data['subscription_expiry'] = datetime.now() + duration
        context.user_data['subscribed'] = True
        await update.message.reply_text(f'Subscription verified! You now have access to all features until {context.user_data["subscription_expiry"].strftime("%Y-%m-%d")}. Use the /start command to begin.')
    else:
        keyboard = [[InlineKeyboardButton("⬅️ Back", callback_data='back')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text('Please subscribe to a plan first. Use the /subscribe command to choose a plan.', reply_markup=reply_markup)

# Define the prediction function
def predict_game(model, server_seed, client_seed, nonce):
    if model is None:
        return "Model not trained for this game."
    # Combine seeds and nonce to create a unique input for the model
    hash_input = f"{server_seed}{client_seed}{nonce}"
    hash_object = hashlib.sha256(hash_input.encode())
    hash_hex = hash_object.hexdigest()
    # Convert hex digest to a numerical format suitable for the model
    numerical_input = [int(hash_hex[i:i+2], 16) for i in range(0, len(hash_hex), 2)]
    # Make a prediction using the trained model
    prediction = model.predict([numerical_input])
    return prediction[0]

# Define a function to check Bitcoin transactions for payment verification
def check_bitcoin_payment(transaction_id, amount):
    # Implement a function to check the Bitcoin blockchain for a transaction
    # This is a placeholder function. Replace it with actual Bitcoin API calls.
    response = requests.get(f'https://api.blockchain.info/rawtx/{transaction_id}')
    if response.status_code == 200:
        tx_data = response.json()
        if tx_data['out'][0]['value'] == amount:
            return True
    return False

# Define a function to check Solana transactions for payment verification
def check_solana_payment(transaction_id, amount):
    # Implement a function to check the Solana blockchain for a transaction
    # This is a placeholder function. Replace it with actual Solana API calls.
    client = Client("https://api.mainnet-beta.solana.com")
    response = client.get_transaction(transaction_id, commitment="confirmed")
    if response['result']['meta']['err'] is None:
        for instruction in response['result']['transaction']['message']['instructions']:
            if instruction['program_id'] == PublicKey(SOLANA_WALLET_ADDRESS):
                if instruction['data'] == amount:
                    return True
    return False

def main():
    import asyncio
    import time
    from telegram.error import Conflict

    # Replace 'YOUR_API_TOKEN_HERE' with your actual Telegram bot API token
    API_TOKEN = '8002162140:AAHDQjBHh-7lOMaZ0qT6V0teLtljTiAp2mY'
    while True:
        try:
            app = Application.builder().token(API_TOKEN).job_queue(None).build()

            app.add_handler(CommandHandler("start", start))
            app.add_handler(CommandHandler("subscribe", subscribe))
            app.add_handler(CallbackQueryHandler(button))
            app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, seed_input))
            app.add_handler(CommandHandler("verify", verify))

            print("Starting bot polling...")
            app.run_polling(drop_pending_updates=True)
        except Conflict as e:
            print(f"Conflict error: {e}. Retrying in 10 seconds...")
            time.sleep(10)
        except Exception as e:
            print(f"Unexpected error: {e}. Retrying in 10 seconds...")
            time.sleep(10)

if __name__ == '__main__':
    main()