from linguistic_variable import LinguisticVariable
from fuzzy_sets import FuzzySet
from fuzzy_control_system import FuzzyControlSystem
from fuzzy_rules import FuzzyRule
from skins import create_sample_skins
import pandas as pd


def create_triangular(a: float, b: float, c: float):
    """Создает треугольную функцию принадлежности."""

    def mu(x: float) -> float:
        if x <= a:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x <= c:
            return (c - x) / (c - b)
        else:
            return 0.0

    return mu


def create_trapezoidal(a: float, b: float, c: float, d: float):
    """Создает трапециевидную функцию принадлежности."""

    def mu(x: float) -> float:
        if x <= a:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x <= c:
            return 1.0
        elif c < x <= d:
            return (d - x) / (d - c)
        else:
            return 0.0

    return mu


def setup_cs_fuzzy_system() -> FuzzyControlSystem:
    """
    Настраивает систему нечеткого управления с полным набором продукционных правил.
    """
    system = FuzzyControlSystem()

    # 1. Степень износа
    wear_var = LinguisticVariable("Wear", 0.0, 1.0, 10000)
    wear_var.add_term(FuzzySet("Factory New", create_triangular(0.0, 0.01, 0.07)))
    wear_var.add_term(
        FuzzySet("Minimal Wear", create_trapezoidal(0.06, 0.08, 0.10, 0.15))
    )
    wear_var.add_term(
        FuzzySet("Field-Tested", create_trapezoidal(0.12, 0.15, 0.30, 0.38))
    )
    wear_var.add_term(
        FuzzySet("Well-Worn", create_trapezoidal(0.35, 0.38, 0.40, 0.45))
    )
    wear_var.add_term(
        FuzzySet("Battle-Scarred", create_triangular(0.40, 0.45, 1.0))
    )
    system.add_input_variable(wear_var)

    # 2. Ликвидность
    liquidity_var = LinguisticVariable("Liquidity", 0, 1000, 1000)
    liquidity_var.add_term(FuzzySet("Very Low", create_triangular(0, 10, 50)))
    liquidity_var.add_term(FuzzySet("Low", create_trapezoidal(30, 100, 150, 300)))
    liquidity_var.add_term(FuzzySet("Medium", create_trapezoidal(200, 300, 400, 600)))
    liquidity_var.add_term(FuzzySet("High", create_trapezoidal(500, 600, 700, 900)))
    liquidity_var.add_term(FuzzySet("Very High", create_triangular(800, 900, 1000)))
    system.add_input_variable(liquidity_var)

    # 3. Рыночная цена
    price_var = LinguisticVariable("Price", 0, 15000, 150000)
    price_var.add_term(FuzzySet("Very Low", create_triangular(0, 0.5, 3)))
    price_var.add_term(FuzzySet("Low", create_trapezoidal(1, 21, 30, 70)))
    price_var.add_term(FuzzySet("Medium", create_trapezoidal(50, 100, 150, 300)))
    price_var.add_term(FuzzySet("High", create_trapezoidal(250, 400, 800, 1500)))
    price_var.add_term(FuzzySet("Very High", create_triangular(1000, 1500, 15000)))
    system.add_input_variable(price_var)

    # 4. Возраст
    max_days = 15 * 365
    age_var = LinguisticVariable("Age", 0.0, float(max_days), num_points=1000)
    age_var.add_term(FuzzySet("New", create_triangular(0.0, 30.0, 180.0)))
    age_var.add_term(FuzzySet("Modern", create_trapezoidal(120.0, 365.0, 730.0, 1095.0)))
    age_var.add_term(FuzzySet("Middle", create_trapezoidal(730.0, 1095.0, 2190.0, 2920.0)))
    age_var.add_term(FuzzySet("Old", create_trapezoidal(2190.0, 2920.0, 3650.0, 4380.0)))
    age_var.add_term(FuzzySet("Vintage", create_triangular(3650.0, 4380.0, float(max_days))))
    system.add_input_variable(age_var)

    # Инвестиционная привлекательность [0.0, 1.0] - Выходная переменная
    investment_var = LinguisticVariable("Investment potential", 0.0, 1.0, 1000)
    investment_var.add_term(FuzzySet("Very Low", create_triangular(0.0, 0.1, 0.3)))
    investment_var.add_term(FuzzySet("Low", create_trapezoidal(0.2, 0.4, 0.5, 0.6)))
    investment_var.add_term(FuzzySet("Medium", create_trapezoidal(0.5, 0.6, 0.7, 0.8)))
    investment_var.add_term(FuzzySet("High", create_trapezoidal(0.7, 0.8, 0.9, 1.0)))
    investment_var.add_term(
        FuzzySet("Very High", create_triangular(0.85, 0.95, 1.0))
    )
    system.add_output_variable(investment_var)

    ## ======= ПРАВИЛА ============

    rules = []

    # 1
    r = FuzzyRule()
    r.add_condition("Wear", "Factory New", "AND")
    r.add_condition("Liquidity", "Very High", "AND")
    r.add_condition("Price", "Very High", "AND")
    r.add_condition("Age", "New", "AND")
    r.set_conclusion("Investment potential", "Very High")
    rules.append(r)

    # 2
    r = FuzzyRule()
    r.add_condition("Wear", "Factory New", "AND")
    r.add_condition("Liquidity", "High", "AND")
    r.add_condition("Price", "High", "AND")
    r.set_conclusion("Investment potential", "High")
    rules.append(r)

    # 3
    r = FuzzyRule()
    r.add_condition("Wear", "Minimal Wear", "AND")
    r.add_condition("Liquidity", "Very High", "AND")
    r.add_condition("Price", "High", "AND")
    r.set_conclusion("Investment potential", "High")
    rules.append(r)

    # 4
    r = FuzzyRule()
    r.add_condition("Age", "Vintage", "AND")
    r.add_condition("Price", "Very High", "AND")
    r.add_condition("Liquidity", "Medium", "AND")
    r.set_conclusion("Investment potential", "Very High")
    rules.append(r)

    # 5
    r = FuzzyRule()
    r.add_condition("Wear", "Field-Tested", "AND")
    r.add_condition("Liquidity", "Medium", "AND")
    r.add_condition("Price", "Medium", "AND")
    r.set_conclusion("Investment potential", "Medium")
    rules.append(r)

    # 6
    r = FuzzyRule()
    r.add_condition("Wear", "Factory New", "AND")
    r.add_condition("Liquidity", "Medium", "AND")
    r.add_condition("Price", "Medium", "AND")
    r.set_conclusion("Investment potential", "Medium")
    rules.append(r)

    # 7
    r = FuzzyRule()
    r.add_condition("Age", "Middle", "AND")
    r.add_condition("Price", "Medium", "AND")
    r.add_condition("Liquidity", "Medium", "AND")
    r.set_conclusion("Investment potential", "Medium")
    rules.append(r)

    # 8
    r = FuzzyRule()
    r.add_condition("Price", "Low", "AND")
    r.add_condition("Liquidity", "High", "AND")
    r.set_conclusion("Investment potential", "High")
    rules.append(r)

    # 9
    r = FuzzyRule()
    r.add_condition("Price", "Low", "AND")
    r.add_condition("Liquidity", "Medium", "AND")
    r.set_conclusion("Investment potential", "Medium")
    rules.append(r)

    # 10
    r = FuzzyRule()
    r.add_condition("Price", "Very Low", "AND")
    r.add_condition("Liquidity", "High", "AND")
    r.set_conclusion("Investment potential", "Medium")
    rules.append(r)

    # 11
    r = FuzzyRule()
    r.add_condition("Wear", "Well-Worn", "AND")
    r.add_condition("Price", "High", "AND")
    r.add_condition("Liquidity", "Low", "AND")
    r.set_conclusion("Investment potential", "Low")
    rules.append(r)

    # 12
    r = FuzzyRule()
    r.add_condition("Wear", "Battle-Scarred", "AND")
    r.add_condition("Price", "Very High", "AND")
    r.add_condition("Liquidity", "Very Low", "AND")
    r.set_conclusion("Investment potential", "Very Low")
    rules.append(r)

    # 13
    r = FuzzyRule()
    r.add_condition("Age", "Old", "AND")
    r.add_condition("Price", "High", "AND")
    r.set_conclusion("Investment potential", "High")
    rules.append(r)

    # 14
    r = FuzzyRule()
    r.add_condition("Age", "Old", "AND")
    r.add_condition("Liquidity", "Low", "AND")
    r.set_conclusion("Investment potential", "Medium")
    rules.append(r)

    # 15
    r = FuzzyRule()
    r.add_condition("Age", "Vintage", "AND")
    r.add_condition("Wear", "Factory New", "AND")
    r.set_conclusion("Investment potential", "Very High")
    rules.append(r)

    # 16
    r = FuzzyRule()
    r.add_condition("Liquidity", "Very Low", "AND")
    r.set_conclusion("Investment potential", "Very Low")
    rules.append(r)

    # 17
    r = FuzzyRule()
    r.add_condition("Price", "Very Low", "AND")
    r.add_condition("Liquidity", "Low", "AND")
    r.set_conclusion("Investment potential", "Low")
    rules.append(r)

    # 18
    r = FuzzyRule()
    r.add_condition("Wear", "Battle-Scarred", "AND")
    r.add_condition("Liquidity", "Medium", "AND")
    r.set_conclusion("Investment potential", "Low")
    rules.append(r)

    # 19
    r = FuzzyRule()
    r.add_condition("Wear", "Factory New", "AND")
    r.add_condition("Price", "Low", "AND")
    r.set_conclusion("Investment potential", "High")
    rules.append(r)

    # 20
    r = FuzzyRule()
    r.add_condition("Price", "Very High", "AND")
    r.add_condition("Liquidity", "Very High", "AND")
    r.set_conclusion("Investment potential", "Very High")
    rules.append(r)

    # Добавление всех правил в систему
    for r in rules:
        system.add_rule(r)

    return system


def demonstrate_rule_evaluation(system: FuzzyControlSystem):
    skins = create_sample_skins()
    
    print(f"\nСоздано {len(skins)} скинов:")
    for i, skin in enumerate(skins, 1):
        print(f"{i}. {skin}")

    summary_data = []

    for skin in skins:
        print(f"\nАнализ скина: {skin.name}")
        
        inputs = {
            "Wear": skin.float_value,
            "Liquidity": skin.liquidity,
            "Price": skin.price,
            "SkinAge": skin.age_days
        }
        
        print(f"Параметры скина:")
        print(f"    - Степень износа (float): {skin.float_value:.4f}")
        print(f"    - Ликвидность: {skin.liquidity} транзакций/день")
        print(f"    - Цена: ${skin.price:.2f}")
        print(f"    - Возраст: {skin.age_days} дней ({skin.age_days/365:.1f} лет)")
        
        # Фазификация
        fuzzified = system.fuzzify(inputs)
        print(f"Фазификация входных параметров:")
        
        for var_name, terms in fuzzified.items():
            print(f"    {var_name}:")
            for term_name, degree in terms.items():
                if degree > 0.01:
                    print(f"    - {term_name}: {degree:.3f}")
        
        # Вывод по Мамдани
        result, universe, aggregated = system.infer_mamdani(
            inputs, "Investment potential"
        )
        
        print(f"Результат нечеткого вывода: {result:.3f}")
        
        # Определяем лингвистическое значение
        investment_var = system.output_vars["Investment potential"]
        
        # Находим термы с наибольшей степенью принадлежности
        fuzzified_investment = {}
        for term_name, term_set in investment_var.terms.items():
            degree = term_set.mu(result)
            if degree > 0.01:
                fuzzified_investment[term_name] = degree
        
        if fuzzified_investment:
            sorted_terms = sorted(fuzzified_investment.items(), 
                                 key=lambda x: x[1], reverse=True)
            
            print(f"Лингвистическая интерпретация:")
            for term_name, degree in sorted_terms:
                print(f"    - {term_name}: {degree:.3f}")
        
        summary_data.append({
            "Skin": skin.name,
            "Wear": f"{skin.float_value:.4f}",
            "Liquidity": f"{skin.liquidity:.0f}",
            "Price": f"${skin.price:.2f}",
            "Age": f"{skin.age_days/365:.1f} years",
            "Result": f"{result:.3f}",
        })
        
    df = pd.DataFrame(summary_data)
    
    print(df.to_markdown(index=False))
    
    return df


def main():
    system = setup_cs_fuzzy_system()
    system.print_system_info()

    for var_name, var in system.input_vars.items():
        var.plot_terms(f"Лингвистическая переменная: {var_name}")

    for var_name, var in system.output_vars.items():
        var.plot_terms(f"Лингвистическая переменная: {var_name}")

    demonstrate_rule_evaluation(system)


if __name__ == "__main__":
    main()
