from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from sqlalchemy.orm import Session
from typing import Dict, List
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, FARMER_ADVISOR_PROMPT_TEMPLATE, MARKET_RESEARCHER_PROMPT_TEMPLATE
from models import Farm, Crop, MarketData, AgentInteraction

class BaseAgent:
    def __init__(self, db_session: Session):
        self.llm = Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
        self.embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.db_session = db_session

    def _log_interaction(self, agent_type: str, query: str, response: str, farm_id: int = None):
        interaction = AgentInteraction(
            agent_type=agent_type,
            query=query,
            response=response,
            farm_id=farm_id
        )
        self.db_session.add(interaction)
        self.db_session.commit()

class FarmerAdvisor(BaseAgent):
    def __init__(self, db_session: Session):
        super().__init__(db_session)
        self.tools = [
            Tool(
                name="analyze_soil",
                func=self._analyze_soil_data,
                description="Analyzes soil data to provide recommendations for sustainable farming"
            ),
            Tool(
                name="calculate_water_needs",
                func=self._calculate_water_requirements,
                description="Calculates optimal water requirements for crops"
            ),
            Tool(
                name="suggest_crop_rotation",
                func=self._suggest_crop_rotation,
                description="Suggests optimal crop rotation patterns"
            )
        ]
        self.agent = create_react_agent(self.llm, self.tools, FARMER_ADVISOR_PROMPT_TEMPLATE)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, memory=self.memory)

    def _analyze_soil_data(self, farm_id: int) -> Dict:
        try:
            farm = self.db_session.query(Farm).filter(Farm.id == farm_id).first()
            if not farm:
                raise ValueError(f"Farm with ID {farm_id} not found")
            if not farm.soil_type:
                raise ValueError(f"Soil type not specified for farm {farm_id}")
            
            soil_recommendations = {
            "sandy": ["Add organic matter", "Use drip irrigation", "Consider drought-resistant crops"],
            "clay": ["Improve drainage", "Add organic matter", "Deep tilling when dry"],
            "loam": ["Maintain organic matter", "Regular crop rotation", "Balanced fertilization"],
            "silt": ["Prevent soil compaction", "Cover cropping", "Careful irrigation"]
        }
        
            analysis = {
                "soil_type": farm.soil_type,
                "recommendations": soil_recommendations.get(farm.soil_type.lower(), []),
                "ph_level": "neutral",  # This would come from actual soil testing
                "organic_matter": "medium",  # This would come from actual soil testing
                "nutrient_levels": {
                    "nitrogen": "moderate",
                    "phosphorus": "moderate",
                    "potassium": "moderate"
                }
            }
            return analysis
        except Exception as e:
            return {"error": str(e), "status": "failed"}

    def _calculate_water_requirements(self, crop_id: int) -> Dict:
        try:
            crop = self.db_session.query(Crop).filter(Crop.id == crop_id).first()
            if not crop:
                raise ValueError(f"Crop with ID {crop_id} not found")
            if crop.area is None or float(crop.area) <= 0:
                raise ValueError(f"Invalid crop area for crop {crop_id}")
            
            # Base water requirements in mm per day for different crops
            base_water_needs = {
                "corn": 6.0,
                "wheat": 4.5,
                "rice": 8.0,
                "soybeans": 5.0,
                "potatoes": 5.5
            }
            
            # Get base water need or use average if crop not in database
            base_need = base_water_needs.get(crop.name.lower(), 5.0)
            
            # Calculate daily water requirements based on area
            daily_water_need = base_need * float(crop.area)  # in cubic meters
            
            # Generate weekly watering schedule
            schedule = [
                {"day": i + 1, "amount": daily_water_need}
                for i in range(7)
            ]
            
            return {
                "crop_name": crop.name,
                "water_needs": daily_water_need,
                "weekly_total": daily_water_need * 7,
                "unit": "cubic_meters",
                "schedule": schedule
            }
        except Exception as e:
            return {"error": str(e), "status": "failed"}


    def _suggest_crop_rotation(self, farm_id: int) -> List[str]:
        try:
            if not isinstance(farm_id, int) or farm_id <= 0:
                return [f"Invalid farm ID: {farm_id}. Farm ID must be a positive integer."]

            # Verify farm exists
            farm = self.db_session.query(Farm).filter(Farm.id == farm_id).first()
            if not farm:
                return [f"Farm with ID {farm_id} not found"]

            crops = self.db_session.query(Crop).filter(Crop.farm_id == farm_id).all()
            if not crops:
                return ["No crops found for this farm"]
        
            # Define crop families and their rotation benefits
            crop_families = {
                "legumes": ["soybeans", "peas", "beans"],
                "cereals": ["corn", "wheat", "rice"],
                "root_crops": ["potatoes", "carrots", "beets"],
                "leafy_greens": ["lettuce", "spinach", "kale"]
            }
        
            current_crops = [crop.name.lower() for crop in crops]
            
            # Identify current crop families
            current_families = []
            for family, crops_list in crop_families.items():
                if any(crop in crops_list for crop in current_crops):
                    current_families.append(family)
            
            # Suggest rotation based on current families
            rotation_suggestions = []
            if "cereals" in current_families:
                rotation_suggestions.append("Follow cereals with legumes to fix nitrogen")
            if "root_crops" in current_families:
                rotation_suggestions.append("Follow root crops with leafy greens")
            if "legumes" in current_families:
                rotation_suggestions.append("Follow legumes with heavy feeders like cereals")
            if "leafy_greens" in current_families:
                rotation_suggestions.append("Follow leafy greens with root crops")
            
            # Add general recommendations if no specific matches
            if not rotation_suggestions:
                rotation_suggestions = [
                    "Start with legumes to improve soil nitrogen",
                    "Follow with heavy feeders like corn or wheat",
                    "Then plant root crops",
                    "Finally, grow leafy greens"
                ]
            
            return rotation_suggestions
        except Exception as e:
            return [f"Error generating crop rotation suggestions: {str(e)}"]

class MarketResearcher(BaseAgent):
    def __init__(self, db_session: Session):
        super().__init__(db_session)
        self.tools = [
            Tool(
                name="analyze_market_trends",
                func=self._analyze_market_trends,
                description="Analyzes market trends for agricultural products"
            ),
            Tool(
                name="predict_crop_prices",
                func=self._predict_crop_prices,
                description="Predicts future crop prices based on historical data"
            ),
            Tool(
                name="scrape_market_data",
                func=self._scrape_market_data,
                description="Scrapes real-time market data from agricultural websites"
            )
        ]
        try:
            self.agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=MARKET_RESEARCHER_PROMPT_TEMPLATE
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create market researcher agent: {str(e)}")
            self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=MARKET_RESEARCHER_PROMPT_TEMPLATE
        )
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, memory=self.memory)

    def _analyze_market_trends(self, crop_name: str, region: str) -> Dict:
        try:
            if not crop_name or not region:
                raise ValueError("Crop name and region must be provided")

            # Fetch historical market data
            market_data = self.db_session.query(MarketData).join(Crop).filter(
                Crop.name == crop_name,
                MarketData.region == region
            ).order_by(MarketData.date.desc()).limit(30).all()
            
            # Convert to pandas DataFrame for analysis
            df = pd.DataFrame([
                {
                    'date': data.date,
                    'price': data.price,
                    'demand': data.demand,
                    'supply': data.supply
                } for data in market_data
            ])
            
            # Calculate trends
            if not df.empty:
                try:
                    price_trend = df['price'].diff().mean()
                    demand_trend = df['demand'].diff().mean()
                    supply_trend = df['supply'].diff().mean()
                    
                    # Validate trend calculations
                    if pd.isna([price_trend, demand_trend, supply_trend]).any():
                        raise ValueError("Invalid trend calculations due to insufficient or invalid data")
                    
                    # Market sentiment analysis
                    if price_trend > 0 and demand_trend > 0:
                        sentiment = 'bullish'
                    elif price_trend < 0 and supply_trend > 0:
                        sentiment = 'bearish'
                    else:
                        sentiment = 'neutral'
                    
                    analysis = {
                        'price_trend': f"{price_trend:.2f}",
                        'demand_trend': f"{demand_trend:.2f}",
                        'supply_trend': f"{supply_trend:.2f}",
                        'market_sentiment': sentiment,
                        'data_points': len(df),
                        'status': 'success'
                    }
                except Exception as calc_error:
                    return {
                        'error': f"Error calculating market trends: {str(calc_error)}",
                        'status': 'failed'
                    }
            else:
                return {
                    'error': f"No market data available for {crop_name} in {region}",
                    'status': 'no_data'
                }
            
            return analysis
        except Exception as e:
            return {
                'error': f"Error analyzing market trends: {str(e)}",
                'status': 'failed'
            }

    def _predict_crop_prices(self, crop_name: str, region: str) -> Dict:
        try:
            if not crop_name or not region:
                raise ValueError("Crop name and region must be provided")

            # Get historical price data
            price_data = self.db_session.query(MarketData).join(Crop).filter(
                Crop.name == crop_name,
                MarketData.region == region
            ).order_by(MarketData.date.desc()).all()
            
            if not price_data:
                return {
                    'error': f"No historical price data available for {crop_name} in {region}",
                    'status': 'no_data'
                }
            
            try:
                # Prepare data for prediction
                df = pd.DataFrame([
                    {
                        'date': data.date,
                        'price': data.price,
                        'demand': data.demand,
                        'supply': data.supply
                    } for data in price_data
                ])
                
                # Validate data
                if df['price'].isnull().any() or df['demand'].isnull().any() or df['supply'].isnull().any():
                    raise ValueError("Missing values in price, demand, or supply data")
                
                if len(df) < 2:  # Need at least 2 data points for trend calculation
                    return {
                        'error': f"Insufficient historical data for {crop_name} in {region}",
                        'status': 'insufficient_data'
                    }
                
                # Normalize features
                scaler = StandardScaler()
                features = ['demand', 'supply']
                df[features] = scaler.fit_transform(df[features])
                
                # Calculate trends and impacts
                price_trend = df['price'].diff().mean()
                demand_impact = df['demand'].mean()
                supply_impact = df['supply'].mean()
                
                if pd.isna([price_trend, demand_impact, supply_impact]).any():
                    raise ValueError("Error calculating trends and impacts")
                
                # Calculate predicted price change
                predicted_change = price_trend * (1 + demand_impact - supply_impact)
                current_price = df['price'].iloc[0]
                predicted_price = current_price + predicted_change
                
                # Validate predictions
                if predicted_price < 0:
                    predicted_price = current_price  # Fallback to current price if prediction is negative
                
                prediction = {
                    'current_price': f"{current_price:.2f}",
                    'predicted_price': f"{predicted_price:.2f}",
                    'confidence': 'medium',
                    'factors': {
                        'price_trend': f"{price_trend:.2f}",
                        'demand_impact': f"{demand_impact:.2f}",
                        'supply_impact': f"{supply_impact:.2f}"
                    },
                    'status': 'success'
                }
                
                return prediction
            except ValueError as ve:
                return {
                    'error': f"Error in price prediction calculations: {str(ve)}",
                    'status': 'calculation_error'
                }
            except Exception as calc_error:
                return {
                    'error': f"Unexpected error in price calculations: {str(calc_error)}",
                    'status': 'calculation_error'
                }
        except Exception as e:
            return {
                'error': f"Error predicting crop prices: {str(e)}",
                'status': 'failed'
            }

    def _scrape_market_data(self, crop_name: str, region: str) -> Dict:
        if not crop_name or not region:
            return {
                'error': 'Crop name and region must be provided',
                'status': 'invalid_input'
            }

        try:
            # Example URL (this would be replaced with actual agricultural data sources)
            url = f"https://example-agri-market.com/prices/{crop_name}/{region}"
            
            # Make request with timeout
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
            except requests.Timeout:
                return {
                    'error': 'Request timed out while fetching market data',
                    'status': 'timeout'
                }
            except requests.RequestException as e:
                return {
                    'error': f'Failed to fetch market data: {str(e)}',
                    'status': 'network_error'
                }
            
            # Parse HTML
            try:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract and validate data
                price_elem = soup.find('span', class_='current-price')
                volume_elem = soup.find('span', class_='trading-volume')
                timestamp_elem = soup.find('span', class_='last-updated')
                
                if not all([price_elem, volume_elem, timestamp_elem]):
                    return {
                        'error': 'Required market data elements not found on page',
                        'status': 'parsing_error'
                    }
                
                try:
                    price = float(price_elem.text.strip())
                    volume = float(volume_elem.text.strip())
                    timestamp = timestamp_elem.text.strip()
                    
                    # Validate values
                    if price <= 0 or volume < 0:
                        raise ValueError('Invalid price or volume values')
                    
                except (ValueError, AttributeError) as ve:
                    return {
                        'error': f'Invalid market data format: {str(ve)}',
                        'status': 'data_error'
                    }
                
                # Store in database
                try:
                    market_data = MarketData(
                        crop_id=1,  # This would be properly mapped in production
                        price=price,
                        supply=volume,
                        demand=volume * 0.8,  # Simplified demand calculation
                        region=region
                    )
                    self.db_session.add(market_data)
                    self.db_session.commit()
                except Exception as db_error:
                    self.db_session.rollback()
                    return {
                        'error': f'Failed to store market data: {str(db_error)}',
                        'status': 'database_error'
                    }
                
                return {
                    'price': price,
                    'volume': volume,
                    'timestamp': timestamp,
                    'source': url,
                    'status': 'success'
                }
                
            except Exception as parse_error:
                return {
                    'error': f'Error parsing market data: {str(parse_error)}',
                    'status': 'parsing_error'
                }
            
        except Exception as e:
            return {
                'error': f'Unexpected error processing market data: {str(e)}',
                'status': 'error'
            }