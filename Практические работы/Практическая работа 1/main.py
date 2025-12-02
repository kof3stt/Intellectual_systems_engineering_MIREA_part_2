from skins import Skin, create_sample_skins
from fuzzy_sets import FuzzySet, TriangularFuzzySet, TrapezoidalFuzzySet
from relations import FuzzyRelation
import numpy as np
import matplotlib.pyplot as plt


def create_skin_fuzzy_sets(skins):
    """
    Создаёт нечёткие множества для скинов на основе их параметров.
    """
    # Множество "Низкая степень износа"
    good_condition = {}
    for skin in skins:
        value = 1.0 - skin.float_value
        good_condition[skin.name] = round(value, 3)
    
    # Множество "Высокая ликвидность"
    high_liquidity = {}
    max_liquidity = max(skin.liquidity for skin in skins)
    min_liquidity = min(skin.liquidity for skin in skins)
    for skin in skins:
        if max_liquidity == min_liquidity:
            value = 0.0
        else:
            offset = 0.1
            normalized = (skin.liquidity - min_liquidity) / (max_liquidity - min_liquidity)
            value = offset + (1 - offset) * normalized
        high_liquidity[skin.name] = round(value, 3)
    
    # Множество "Высокая цена"
    high_price = {}
    max_price = max(skin.price for skin in skins)
    min_price = min(skin.price for skin in skins)
    for skin in skins:
        if max_price == min_price:
            value = 0.0
        else:
            offset = 0.1
            normalized = (skin.price - min_price) / (max_price - min_price)
            value = offset + (1 - offset) * normalized
        high_price[skin.name] = round(value, 3)
    
    # Множество "Старые скины"
    old_skins = {}
    max_age = max(skin.age_days for skin in skins)
    min_age = min(skin.age_days for skin in skins)
    for skin in skins:
        if max_age == min_age:
            value = 0.5
        else:
            offset = 0.1
            normalized = (skin.age_days - min_age) / (max_age - min_age)
            value = offset + (1 - offset) * normalized
        old_skins[skin.name] = round(value, 3)
    
    # Множество "Инвестиционная привлекательность" 
    investment = {}
    for skin in skins:
        condition_score = 1.0 - skin.float_value
        price_score = min(skin.price / 10000.0, 1.0)
        liquidity_score = min(skin.liquidity / 100.0, 1.0)
        value = (condition_score + price_score + liquidity_score) / 3.0
        investment[skin.name] = round(value, 2)
    
    return {
        "Низкая степень износа": FuzzySet("Низкая степень износа", good_condition),
        "Высокая ликвидность": FuzzySet("Высокая ликвидность", high_liquidity),
        "Высокая цена": FuzzySet("Высокая цена", high_price),
        "Старые скины": FuzzySet("Старые скины", old_skins),
        "Инвестиционная привлекательность": FuzzySet("Инвестиционная привлекательность", investment),
    }


def build_relation(set_a: FuzzySet, set_b: FuzzySet) -> FuzzyRelation:
    """Строит продукционное отношение между двумя нечёткими множествами."""
    elements_a = set_a.get_elements()
    elements_b = set_b.get_elements()
    
    matrix = []
    for elem_a in elements_a:
        row = []
        mu_a = set_a.mu(elem_a)
        for elem_b in elements_b:
            mu_b = set_b.mu(elem_b)
            row.append(min(mu_a, mu_b))
        matrix.append(row)
    
    return FuzzyRelation(elements_a, elements_b, matrix, 
                        f"{set_a.name} → {set_b.name}")


def plot_relation(relation: FuzzyRelation, title = None):
    """Визуализация отношения как тепловой карты."""
    matrix = np.array(relation.matrix)
    rows = relation.rows
    cols = relation.columns
    
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    plt.colorbar(label='μ')
    
    plt.xticks(range(len(cols)), cols, rotation=45, ha='right')
    plt.yticks(range(len(rows)), rows)
    
    plt.title(title if title is not None else relation.name)
    
    for i in range(len(rows)):
        for j in range(len(cols)):
            plt.text(j, i, f'{matrix[i, j]:.2f}', 
                    ha='center', va='center', 
                    color='black' if matrix[i, j] < 0.7 else 'white')
    
    plt.tight_layout()
    plt.show()


def compose_relations(r1: FuzzyRelation, r2: FuzzyRelation) -> FuzzyRelation:
    """Композиция двух отношений (max-min композиция)."""
    if r1.columns != r2.rows:
        raise ValueError("Несовместимые отношения для композиции")
    
    n = len(r1.rows)
    m = len(r2.columns)
    p = len(r1.columns)
    
    result_matrix = []
    for i in range(n):
        row = []
        for j in range(m):
            max_val = 0.0
            for k in range(p):
                val = min(r1.matrix[i][k], r2.matrix[k][j])
                max_val = max(max_val, val)
            row.append(max_val)
        result_matrix.append(row)
    
    return FuzzyRelation(r1.rows, r2.columns, result_matrix,
                        f"({r1.name}) ∘ ({r2.name})")


def main():
    # Создание объектов скинов
    skins = create_sample_skins()
    print(f"Создано {len(skins)} скинов:")
    for skin in skins:
        print(f"   - {skin}")
    
    # Построение множеств
    fuzzy_sets = create_skin_fuzzy_sets(skins)
    print(f"\nСоздано {len(fuzzy_sets)} нечётких множеств:")
    for name, fset in fuzzy_sets.items():
        print(f"   - {name}: {dict(zip(fset.get_elements(), fset.get_values()))}")
    
    # Визуализация множеств
    print("\nВизуализация нечётких множеств:")
    for name, fset in fuzzy_sets.items():
        print(f"   - Визуализация {name}")
        fset.plot()
    
    # Построение отношений
    print("\nПостроение продукционных отношений:")
    
    # Отношение 1: Низкая степень износа → Высокая цена
    relation1 = build_relation(fuzzy_sets["Низкая степень износа"], 
                              fuzzy_sets["Высокая цена"])
    print(f"   - {relation1.name}")
    plot_relation(relation1)
    
    # Отношение 2: Высокая цена → Инвестиционная привлекательность
    relation2 = build_relation(fuzzy_sets["Высокая цена"], 
                              fuzzy_sets["Инвестиционная привлекательность"])
    print(f"   - {relation2.name}")
    plot_relation(relation2)

    # Отношение 3: Высокая ликвидность → Инвестиционная привлекательность
    relation3 = build_relation(fuzzy_sets["Высокая ликвидность"], 
                              fuzzy_sets["Инвестиционная привлекательность"])
    print(f"   - {relation3.name}")
    plot_relation(relation3)
    
    # Отношение 4: Старые скины → Высокая цена
    relation4 = build_relation(fuzzy_sets["Старые скины"], 
                              fuzzy_sets["Высокая цена"])
    print(f"   - {relation4.name}")
    plot_relation(relation4)

    # Композиция отношений
    print("\nКомпозиция отношений:")
    print(f"   - ({relation1.name}) ∘ ({relation2.name})")
    composition = compose_relations(relation1, relation2)
    plot_relation(composition)

    # Транспонирование
    print("\nТранспонирование отношения:")
    relation1_transposed = relation1.transpose()
    print(f"   - Транспонированное: {relation1_transposed.name}")
    plot_relation(relation1_transposed, "Высокая цена → Низкая степень износа")
    
    # Дополнение множеств
    print("\nДополнение нечётких множеств:")
    for name, fset in list(fuzzy_sets.items())[:1]:
        complement_set = fset.complement()
        print(f"   - Дополнение {name}: {complement_set.name}")
        complement_set.plot()
    
    # Правила вывода (modus ponens)
    print("\n8. Применение правила вывода (modus ponens):")
    
    # Создаём новое множество "Очень низкая степень износа"
    very_good_condition = {}
    for skin in skins:
        value = (1.0 - skin.float_value) ** 2
        very_good_condition[skin.name] = round(value, 2)
    
    fact_set = FuzzySet("Очень низкая степень износа", very_good_condition)
    print(f"   - Факт: {fact_set.name}")
    fact_set.plot()
    
    # Применяем правило modus ponens
    print(f"   - Применяем правило: {fact_set.name} → {relation1.name}")
    result = fact_set.apply_modus_ponens(relation1)
    if result:
        print(f"   - Результат вывода: {result.name}")
        result.plot()


if __name__ == "__main__":
    main()
