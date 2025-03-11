# Setting Up Binance Testnet API Keys

To use this trading bot with Binance Testnet, you need to create API keys for testing. This guide will walk you through the process.

## Step 1: Create a Binance Account

If you don't already have a Binance account, you'll need to create one:

1. Go to [Binance.com](https://www.binance.com/)
2. Click "Register" and follow the instructions to create an account
3. Complete the identity verification process if required

## Step 2: Access the Binance Testnet

The Binance Testnet is a separate environment for testing:

1. Go to [Binance Testnet](https://testnet.binance.vision/)
2. Click "Log In with GitHub" (this is the easiest method)
3. Authorize the application to access your GitHub account

## Step 3: Generate API Keys

1. After logging in, click on "Generate HMAC_SHA256 Key"
2. The system will generate an API key and secret key pair
3. **IMPORTANT**: Save both the API Key and Secret Key immediately! They will only be shown once.

Your keys will look something like this:

- API Key: `vmPUZE6mv9SD5VNHk4HlWFsOr6aKE2zvsw0MuIgwCIPy6utIco14y7Ju91duEh8A`
- Secret Key: `NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j`

## Step 4: Configure the Trading Bot

1. Copy the `config.ini.example` file to `config.ini`:

   ``` plaintext
   cp config.ini.example config.ini

   ```

2. Edit the `config.ini` file and replace the placeholders with your actual keys:

   ``` plaintext
   [binance]
   api_key = YOUR_TESTNET_API_KEY_HERE
   api_secret = YOUR_TESTNET_SECRET_KEY_HERE
   use_testnet = True
   ```

3. Save the file and run the trading bot

## Required API Permissions

The API keys generated from Binance Testnet should automatically have the following permissions:

1. **Read Information** - Required to fetch market data and account information
2. **Enable Trading** - Required to place and manage orders
3. **Enable Withdrawals** - Not required for this bot

If you're getting authentication errors, the most likely causes are:

- **Invalid API Key Format** - Make sure you've copied the entire key without any extra spaces
- **Expired API Key** - Testnet keys expire periodically; generate new ones if needed
- **IP Restriction** - If you added IP restrictions, make sure your current IP is allowed
- **Wrong Environment** - Ensure you're using testnet keys with `use_testnet = True`

## Troubleshooting API Errors

### "Invalid API-key, IP, or permissions for action"

This error typically indicates one of the following issues:

1. **Expired API Key**: Testnet API keys are temporary and expire after some time
   - Solution: Generate a new API key pair on the Testnet site

2. **Wrong Key Format**: The key wasn't copied correctly
   - Solution: Regenerate and carefully copy the full key without spaces

3. **Missing Permissions**: The key doesn't have required permissions
   - Solution: Regenerate the key on Testnet with proper permissions

4. **IP Restriction**: Your IP is not on the allowed list
   - Solution: Remove IP restrictions or add your current IP

### Checking Your API Key Status

You can check if your API key is valid by:

1. Logging into the Binance Testnet dashboard
2. Looking for your existing keys (if they're listed, they're still valid)
3. If you don't see your keys or they show as expired, generate new ones

## Important Notes

- Testnet API keys are temporary and may expire after some time
- If you encounter authentication errors, generate new keys
- The testnet environment is reset periodically, so your test funds may disappear
- Never use your real Binance API keys for testing
- Keep your API keys secure and never share them with anyone
- Configure IP restrictions for your API keys when using the real Binance API
