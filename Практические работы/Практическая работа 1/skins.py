from random import randint


class Skin:
    """
    Класс для представления конкретного скина с его параметрами.
    """
    
    def __init__(self, name: str, float_value: float, liquidity: float, 
                 price: float, age_days: float, paint_seed: int = 0):
        self.name = name
        self.float_value = float_value
        self.liquidity = liquidity
        self.price = price
        self.age_days = age_days
        self.paint_seed = paint_seed
        
    def __str__(self):
        return f"{self.name}: float={self.float_value:.3f}, price=${self.price}, age={self.age_days} days"
    
    def __repr__(self):
        return f"Skin('{self.name}')"


def create_sample_skins():
    """Создает список скинов для демонстрации"""
    return [
        Skin(name="AK-47 | Redline", float_value=0.156, liquidity=127, price=32.99, age_days=4302, paint_seed=randint(1, 999)),
        Skin(name="AWP | Dragon Lore", float_value=0.035, liquidity=1, price=11850, age_days=4171, paint_seed=randint(1, 999)),
        Skin(name="M4A1-S | Hyper Beast", float_value=0.366, liquidity=30, price=128.75, age_days=3883, paint_seed=randint(1, 999)),
        Skin(name="Karambit | Fade", float_value=0.0102, liquidity=5, price=2350, age_days=4492, paint_seed=randint(1, 999)),
        Skin(name="Sport Gloves | Vice", float_value=0.09, liquidity=3, price=2400, age_days=2846, paint_seed=randint(1, 999))
    ]
