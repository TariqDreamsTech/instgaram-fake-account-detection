# Apify Setup Guide for Instagram Profile Fetcher

## Overview

The Instagram profile fetcher uses **Apify API exclusively** for reliable data extraction. Apify handles Instagram's anti-scraping measures and provides accurate profile data.

## Setup Instructions

### 1. Get Your Apify API Token

1. Sign up for a free account at [https://apify.com](https://apify.com)
2. Go to your [Settings](https://console.apify.com/account/integrations)
3. Copy your API token

### 2. Set Environment Variable

You can set the API token in one of the following ways:

#### Option A: Environment Variable (Recommended)
```bash
export APIFY_API_TOKEN="your_api_token_here"
```

#### Option B: In your shell profile (persistent)
Add to `~/.bashrc` or `~/.zshrc`:
```bash
export APIFY_API_TOKEN="your_api_token_here"
```

#### Option C: In your Python code (before running the app)
```python
import os
os.environ["APIFY_API_TOKEN"] = "your_api_token_here"
```

### 3. Install Dependencies

```bash
pip install apify-client
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## How It Works

The fetcher uses **Apify API only**:
- Reliable and accurate data extraction
- Works with private profiles (if you have access)
- Handles rate limiting automatically
- No HTML parsing or GraphQL queries needed

## Usage

The fetcher requires:
1. Apify client installed (`pip install apify-client`)
2. API token set as environment variable (`APIFY_API_TOKEN`)

No code changes needed - just set the API token and it will work!

## Apify Actor Used

- **Actor ID**: `dSCLg0C3YEZ83HzYX`
- **Actor Name**: Instagram Profile Scraper
- **Input**: Username(s)
- **Output**: Profile data including posts, followers, following, biography, etc.

## Cost Considerations

- Apify has a free tier with limited usage
- Check [Apify Pricing](https://apify.com/pricing) for details
- Each profile fetch consumes Apify compute units

## Troubleshooting

### "APIFY_API_TOKEN not set" error
- Make sure you've set the environment variable correctly
- Restart your terminal/application after setting the variable
- Verify with: `echo $APIFY_API_TOKEN`

### Apify returns empty results
- Check if the profile is accessible
- Verify your Apify account has credits
- Check the Apify actor status at https://console.apify.com

### Profile fetch fails
- Profile might be private and inaccessible
- Check if the username/URL is correct
- Verify your Apify account has sufficient credits
- Check Apify console for any error messages

