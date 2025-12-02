from typing import List


class FuzzyRelation:
    """
    Класс для представления нечёткого отношения между двумя множествами.
    """
    
    def __init__(self, rows: List[str], columns: List[str], 
                 matrix: List[List[float]], name: str = "Неименованное отношение"):
        """
        Args:
            rows: имена элементов для строк (посылка)
            columns: имена элементов для столбцов (заключение)
            matrix: матрица отношения
            name: имя отношения
        """
        self.rows = rows
        self.columns = columns
        self.matrix = matrix
        self.name = name
        
    def __str__(self):
        return f"FuzzyRelation: {self.name} ({len(self.rows)}x{len(self.columns)})"
    
    def transpose(self) -> 'FuzzyRelation':
        """Транспонирование отношения (меняет посылку и заключение местами)."""
        transposed_matrix = []
        for j in range(len(self.columns)):
            row = []
            for i in range(len(self.rows)):
                row.append(self.matrix[i][j])
            transposed_matrix.append(row)
        
        return FuzzyRelation(
            rows=self.columns,
            columns=self.rows,
            matrix=transposed_matrix,
            name=f"Транспонированное: {self.name}"
        )
    
    def complement(self) -> 'FuzzyRelation':
        """Дополнение отношения (1 - значение)."""
        complemented_matrix = []
        for i in range(len(self.rows)):
            row = [1.0 - val for val in self.matrix[i]]
            complemented_matrix.append(row)
        
        return FuzzyRelation(
            rows=self.rows,
            columns=self.columns,
            matrix=complemented_matrix,
            name=f"Дополнение: {self.name}"
        )
