from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import List, Optional
from pydantic import BaseModel
import uvicorn

# Initialize templates
templates = Jinja2Templates(directory="templates")

from config import DATABASE_URL, API_HOST, API_PORT
from models import Base, Farm, Crop, MarketData
from agents import FarmerAdvisor, MarketResearcher

# Initialize FastAPI app
app = FastAPI(title="Sustainable Farming AI System")

# Database setup
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models for request/response
class FarmBase(BaseModel):
    name: str
    location: str
    soil_type: Optional[str] = None
    total_area: Optional[float] = None

class CropBase(BaseModel):
    name: str
    planting_date: Optional[str] = None
    harvest_date: Optional[str] = None
    area: Optional[float] = None
    expected_yield: Optional[float] = None

class AdvisorQuery(BaseModel):
    farm_id: int
    query: str

class MarketQuery(BaseModel):
    crop_name: str
    region: str
    query: str

# API endpoints
@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/farms/")
async def create_farm(farm: FarmBase, db: Session = Depends(get_db)):
    db_farm = Farm(**farm.dict())
    db.add(db_farm)
    db.commit()
    db.refresh(db_farm)
    return db_farm

@app.post("/farms/{farm_id}/crops/")
async def add_crop(farm_id: int, crop: CropBase, db: Session = Depends(get_db)):
    farm = db.query(Farm).filter(Farm.id == farm_id).first()
    if not farm:
        raise HTTPException(status_code=404, detail=f"Farm with ID {farm_id} not found")
    
    try:
        crop_data = crop.dict()
        
        # Convert date strings to datetime objects if provided
        from datetime import datetime
        if crop_data.get('planting_date'):
            try:
                crop_data['planting_date'] = datetime.strptime(crop_data['planting_date'], '%Y-%m-%d')
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid planting_date format. Use YYYY-MM-DD")
        
        if crop_data.get('harvest_date'):
            try:
                crop_data['harvest_date'] = datetime.strptime(crop_data['harvest_date'], '%Y-%m-%d')
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid harvest_date format. Use YYYY-MM-DD")
        
        # Validate area and expected_yield if provided
        if crop_data.get('area') is not None and crop_data['area'] <= 0:
            raise HTTPException(status_code=400, detail="Area must be greater than 0")
        
        if crop_data.get('expected_yield') is not None and crop_data['expected_yield'] <= 0:
            raise HTTPException(status_code=400, detail="Expected yield must be greater than 0")
        
        # Create and save the crop
        db_crop = Crop(**crop_data, farm_id=farm_id)
        db.add(db_crop)
        db.commit()
        db.refresh(db_crop)
        return db_crop
    except HTTPException as he:
        db.rollback()
        raise he
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/advisor/query/")
async def query_advisor(query: AdvisorQuery, db: Session = Depends(get_db)):
    advisor = FarmerAdvisor(db)
    response = advisor.agent_executor.run(
        context=f"Farm ID: {query.farm_id}",
        question=query.query
    )
    return {"response": response}

@app.post("/market/query/")
async def query_market(query: MarketQuery, db: Session = Depends(get_db)):
    researcher = MarketResearcher(db)
    response = researcher.agent_executor.run(
        context=f"Crop: {query.crop_name}, Region: {query.region}",
        question=query.query
    )
    return {"response": response}

@app.get("/farms/{farm_id}/recommendations/")
async def get_farm_recommendations(farm_id: int, db: Session = Depends(get_db)):
    farm = db.query(Farm).filter(Farm.id == farm_id).first()
    if not farm:
        raise HTTPException(status_code=404, detail=f"Farm with ID {farm_id} not found")
    
    try:
        advisor = FarmerAdvisor(db)
        soil_analysis = advisor._analyze_soil_data(farm_id)
        if "error" in soil_analysis:
            raise HTTPException(status_code=400, detail=soil_analysis["error"])
            
        crops = db.query(Crop).filter(Crop.farm_id == farm_id).all()
        water_requirements = []
        if not crops:
            water_requirements = ["No crops found for this farm"]
        else:
            for crop in crops:
                try:
                    water_req = advisor._calculate_water_requirements(crop.id.real)
                    if isinstance(water_req, dict) and "error" not in water_req:
                        water_requirements.append(water_req)
                except Exception as e:
                    continue  # Skip failed calculations and continue with next crop
                
        recommendations = {
            "soil_analysis": soil_analysis,
            "crop_rotation": advisor._suggest_crop_rotation(farm_id),
            "water_requirements": water_requirements
        }
        return recommendations
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/analysis/{crop_name}")
async def get_market_analysis(crop_name: str, region: str, db: Session = Depends(get_db)):
    researcher = MarketResearcher(db)
    analysis = {
        "market_trends": researcher._analyze_market_trends(crop_name, region),
        "price_prediction": researcher._predict_crop_prices(crop_name, region)
    }
    return analysis

if __name__ == "__main__":
    uvicorn.run("app:app", host=API_HOST, port=API_PORT, reload=True)