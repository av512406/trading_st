VWAP crossover bot (trading_setup)

Files in this folder:
- .env.example     -> template for environment variables (copy to .env and edit)
- config.example.yaml -> optional YAML config example
- vwap_bot.py      -> Python skeleton for VWAP+SMA calculated_line crossover strategy
- requirements.txt -> python dependencies

Quick start (local dry-run):
1. Copy `.env.example` to `.env` in this folder and set `DRY_RUN=true`.
2. Create a venv and install dependencies:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Test on historical candles:

   ```powershell
   python vwap_bot.py --candles path\to\hourly_candles.csv --once
   ```

Notes:
- Fill `DELTA_API_KEY` and `DELTA_API_SECRET` in a secure way before running live. Use AWS Secrets Manager or systemd EnvironmentFile with secure permissions on EC2.
- The broker methods in `vwap_bot.py` are stubs. Replace them with Delta Exchange REST API calls (and follow their HMAC signing requirements) before going live.
- Start with `DRY_RUN=true` and test on sandbox/testnet if available.
