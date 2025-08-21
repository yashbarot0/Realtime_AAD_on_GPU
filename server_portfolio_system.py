#!/usr/bin/env python3
"""
Enhanced Real-Time Portfolio System for Linux Server Deployment
Optimized for high-performance computing with GPU acceleration
"""

import asyncio
import json
import argparse
import signal
import sys
import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import psutil
import time
from pathlib import Path

# Try to import optional server components
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    SERVER_MODE_AVAILABLE = True
except ImportError:
    SERVER_MODE_AVAILABLE = False
    print("⚠️  FastAPI not available. Server mode disabled.")
    print("   Install with: pip install fastapi uvicorn")

# Import our core components
from realtime_portfolio_system import RealTimePortfolioSystem

class ServerPortfolioSystem:
    """Enhanced portfolio system for server deployment"""
    
    def __init__(self, config_file: str = "production_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
        self.portfolio_system = None
        self.app = None
        self.is_running = False
        
        # Performance tracking
        self.stats = {
            'start_time': None,
            'total_computations': 0,
            'total_updates': 0,
            'avg_latency_ms': 0.0,
            'last_update': None,
            'gpu_utilization': 0.0,
            'memory_usage': 0.0
        }
        
        # Setup logging
        self.setup_logging()
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            print(f"✅ Loaded configuration from {self.config_file}")
            return config
        except FileNotFoundError:
            print(f"⚠️  Config file {self.config_file} not found. Using defaults.")
            return self.get_default_config()
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing config file: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "server_mode": True,
            "market_data": {
                "update_interval_seconds": 30,
                "symbols": ["AAPL", "MSFT", "GOOGL", "TSLA"],
                "data_source": "yfinance"
            },
            "compute_config": {
                "update_frequency_ms": 1000,
                "use_gpu": True,
                "max_positions": 1000
            },
            "logging": {
                "enable_console": True,
                "enable_file": True,
                "log_level": "INFO"
            },
            "api": {
                "enable_rest_api": True,
                "port": 8080,
                "host": "0.0.0.0"
            }
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('log_level', 'INFO'))
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup root logger
        logger = logging.getLogger()
        logger.setLevel(log_level)
        
        # Console handler
        if log_config.get('enable_console', True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if log_config.get('enable_file', False):
            log_file = log_config.get('log_file', 'portfolio_greeks.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        self.logger = logging.getLogger(__name__)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}. Shutting down gracefully...")
        asyncio.create_task(self.shutdown())
    
    async def initialize(self):
        """Initialize the portfolio system"""
        try:
            # Get symbols from config
            symbols = self.config['market_data']['symbols']
            update_interval = self.config['market_data']['update_interval_seconds']
            
            # Create portfolio system
            self.portfolio_system = RealTimePortfolioSystem(symbols, update_interval)
            
            # Load portfolio positions if file exists
            await self.load_portfolio_positions()
            
            # Set up callbacks
            self.portfolio_system.set_greeks_callback(self.on_greeks_update)
            self.portfolio_system.set_alert_callback(self.on_risk_alert)
            
            # Initialize REST API if enabled
            if self.config.get('api', {}).get('enable_rest_api', False):
                await self.setup_rest_api()
            
            self.logger.info("Portfolio system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize portfolio system: {e}")
            return False
    
    async def load_portfolio_positions(self):
        """Load portfolio positions from file"""
        portfolio_files = ['large_portfolio_sample.json', 'sample_portfolio.json']
        
        for portfolio_file in portfolio_files:
            if Path(portfolio_file).exists():
                try:
                    with open(portfolio_file, 'r') as f:
                        portfolio_data = json.load(f)
                    
                    positions = portfolio_data.get('positions', [])
                    for pos in positions:
                        self.portfolio_system.add_position(
                            pos['symbol'],
                            pos['option_type'],
                            pos['strike'],
                            pos['expiration'],
                            pos['quantity'],
                            pos.get('premium_paid', 0.0)
                        )
                    
                    self.logger.info(f"Loaded {len(positions)} positions from {portfolio_file}")
                    return
                    
                except Exception as e:
                    self.logger.error(f"Error loading portfolio from {portfolio_file}: {e}")
        
        self.logger.warning("No portfolio file found. Starting with empty portfolio.")
    
    async def setup_rest_api(self):
        """Setup REST API endpoints"""
        if not SERVER_MODE_AVAILABLE:
            self.logger.warning("FastAPI not available. REST API disabled.")
            return
        
        self.app = FastAPI(title="Portfolio Greeks API", version="1.0.0")
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptime_seconds": time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
            }
        
        # Get current portfolio Greeks
        @self.app.get("/portfolio/greeks")
        async def get_portfolio_greeks():
            if not self.portfolio_system:
                raise HTTPException(status_code=503, detail="Portfolio system not initialized")
            
            greeks = self.portfolio_system.get_current_greeks()
            if not greeks:
                raise HTTPException(status_code=404, detail="No Greeks data available")
            
            return greeks
        
        # Get performance statistics
        @self.app.get("/stats")
        async def get_stats():
            # Update system stats
            self.update_system_stats()
            return self.stats
        
        # Add position endpoint
        @self.app.post("/portfolio/positions")
        async def add_position(position: dict):
            try:
                pos_id = self.portfolio_system.add_position(
                    position['symbol'],
                    position['option_type'],
                    position['strike'],
                    position['expiration'],
                    position['quantity'],
                    position.get('premium_paid', 0.0)
                )
                return {"position_id": pos_id, "status": "added"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        # Get all positions
        @self.app.get("/portfolio/positions")
        async def get_positions():
            if not self.portfolio_system:
                raise HTTPException(status_code=503, detail="Portfolio system not initialized")
            
            positions = self.portfolio_system.portfolio_manager.get_positions()
            return {"positions": positions}
        
        # Remove position endpoint
        @self.app.delete("/portfolio/positions/{position_id}")
        async def remove_position(position_id: int):
            if not self.portfolio_system:
                raise HTTPException(status_code=503, detail="Portfolio system not initialized")
            
            success = self.portfolio_system.remove_position(position_id)
            if success:
                return {"status": "removed"}
            else:
                raise HTTPException(status_code=404, detail="Position not found")
        
        self.logger.info("REST API endpoints configured")
    
    async def on_greeks_update(self, greeks: Dict):
        """Handle Greeks updates"""
        self.stats['total_computations'] += 1
        self.stats['last_update'] = datetime.now(timezone.utc).isoformat()
        
        if 'computation_time_ms' in greeks:
            # Update average latency
            current_avg = self.stats['avg_latency_ms']
            count = self.stats['total_computations']
            new_latency = greeks['computation_time_ms']
            
            if count == 1:
                self.stats['avg_latency_ms'] = new_latency
            else:
                self.stats['avg_latency_ms'] = (current_avg * (count - 1) + new_latency) / count
        
        # Log summary
        self.logger.info(
            f"Portfolio update: Delta={greeks.get('total_delta', 0):.2f}, "
            f"Exposure=${greeks.get('total_exposure', 0):,.2f}, "
            f"Positions={greeks.get('num_positions', 0)}, "
            f"Latency={greeks.get('computation_time_ms', 0):.2f}ms"
        )
    
    async def on_risk_alert(self, alert: Dict):
        """Handle risk alerts"""
        self.logger.warning(f"RISK ALERT: {alert.get('message', 'Unknown alert')}")
        
        # In production, you might want to:
        # - Send email/SMS notifications
        # - Write to a separate alert log
        # - Trigger automated hedging
        # - Update dashboard alerts
    
    def update_system_stats(self):
        """Update system performance statistics"""
        # Memory usage
        process = psutil.Process()
        self.stats['memory_usage'] = process.memory_info().rss / 1024 / 1024  # MB
        
        # GPU utilization (if available)
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.stats['gpu_utilization'] = float(result.stdout.strip())
        except:
            pass  # GPU stats not available
    
    async def start(self):
        """Start the portfolio system"""
        self.logger.info("Starting Real-Time Portfolio Greeks Server...")
        self.stats['start_time'] = time.time()
        self.is_running = True
        
        # Initialize system
        if not await self.initialize():
            self.logger.error("Failed to initialize. Exiting.")
            return False
        
        # Start portfolio system
        portfolio_task = asyncio.create_task(self.portfolio_system.start())
        
        # Start REST API server if enabled
        api_task = None
        if self.app and self.config.get('api', {}).get('enable_rest_api', False):
            api_config = self.config['api']
            host = api_config.get('host', '0.0.0.0')
            port = api_config.get('port', 8080)
            
            config = uvicorn.Config(self.app, host=host, port=port, log_level="info")
            server = uvicorn.Server(config)
            api_task = asyncio.create_task(server.serve())
            
            self.logger.info(f"REST API server starting on http://{host}:{port}")
        
        try:
            # Run until shutdown
            if api_task:
                await asyncio.gather(portfolio_task, api_task)
            else:
                await portfolio_task
                
        except asyncio.CancelledError:
            self.logger.info("Tasks cancelled during shutdown")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        
        return True
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down portfolio system...")
        self.is_running = False
        
        if self.portfolio_system:
            await self.portfolio_system.stop()
        
        # Cancel all running tasks
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Final stats
        if self.stats['start_time']:
            runtime = time.time() - self.stats['start_time']
            self.logger.info(f"Total runtime: {runtime:.1f} seconds")
            self.logger.info(f"Total computations: {self.stats['total_computations']}")
            self.logger.info(f"Average latency: {self.stats['avg_latency_ms']:.2f} ms")
        
        self.logger.info("Shutdown complete")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Real-Time Portfolio Greeks Server")
    parser.add_argument('--config', default='production_config.json', 
                       help='Configuration file path')
    parser.add_argument('--server-mode', action='store_true',
                       help='Run in server mode with REST API')
    parser.add_argument('--demo-mode', action='store_true',
                       help='Run in demo mode for testing')
    
    args = parser.parse_args()
    
    if args.demo_mode:
        print("🧪 Running in demo mode...")
        # Use the original demo from realtime_portfolio_system.py
        from realtime_portfolio_system import demo_realtime_system
        await demo_realtime_system()
        return
    
    # Create and start server
    server = ServerPortfolioSystem(args.config)
    
    try:
        success = await server.start()
        if not success:
            print("❌ Failed to start server")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
    finally:
        await server.shutdown()

if __name__ == "__main__":
    print("🚀 Real-Time Portfolio Greeks - Linux Server Edition")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        sys.exit(1)
    
    # Check if running on Linux
    if sys.platform.startswith('linux'):
        print("✅ Running on Linux server")
    else:
        print("⚠️  Not running on Linux - some optimizations may not work")
    
    asyncio.run(main())
