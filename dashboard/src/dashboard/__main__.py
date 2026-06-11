"""Entry point for `python -m dashboard`."""
import asyncio
from dashboard.api import main

asyncio.run(main())
