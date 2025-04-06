from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Farm(Base):
    __tablename__ = 'farms'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    location = Column(String(200), nullable=False)
    soil_type = Column(String(50))
    total_area = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    crops = relationship('Crop', back_populates='farm')

class Crop(Base):
    __tablename__ = 'crops'
    
    id = Column(Integer, primary_key=True)
    farm_id = Column(Integer, ForeignKey('farms.id'))
    name = Column(String(100), nullable=False)
    planting_date = Column(DateTime)
    harvest_date = Column(DateTime)
    area = Column(Float)
    expected_yield = Column(Float)
    actual_yield = Column(Float)
    water_usage = Column(Float)
    pesticide_usage = Column(Float)
    
    farm = relationship('Farm', back_populates='crops')
    market_data = relationship('MarketData', back_populates='crop')

class MarketData(Base):
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    crop_id = Column(Integer, ForeignKey('crops.id'))
    date = Column(DateTime, default=datetime.utcnow)
    price = Column(Float)
    demand = Column(Float)
    supply = Column(Float)
    region = Column(String(100))
    
    crop = relationship('Crop', back_populates='market_data')

class AgentInteraction(Base):
    __tablename__ = 'agent_interactions'
    
    id = Column(Integer, primary_key=True)
    agent_type = Column(String(50))  # 'farmer_advisor' or 'market_researcher'
    query = Column(Text)
    response = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    farm_id = Column(Integer, ForeignKey('farms.id'))
    context_data = Column(Text)  # Store additional context as JSON