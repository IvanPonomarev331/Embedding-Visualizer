# Контракт датасета, версия 1

Один набор описывает совместимое векторное пространство и один снимок данных.
Совместимость обеспечивает поставщик: одинаковая размерность не доказывает,
что векторы разных моделей можно сопоставлять.

## Manifest

Обязательны непустые строки `dataset_name`, `space_id`, `version` и положительное
целое `embedding_dim`. Значения по умолчанию: `schema_version=1`,
`default_metric="cosine"`. Допустимы также `euclidean` и `dot_product`.
Неизвестные поля и версии схемы отвергаются.

## Таблица

- `id`: уникальная непустая строка. Пробелы и ведущие нули не нормализуются.
  Для состояний пользователя ID должен различать снимки; исходный user ID
  и время можно хранить в metadata.
- `embedding`: одномерный числовой вектор длины `embedding_dim`. Float64 и
  целые числа преобразуются в float32. Запрещены bool, комплексные числа,
  строки, null, NaN, Inf и переполнение float32. Нормализация не применяется.
- `name`: необязательная строка или null.
- `metadata`: необязательный JSON-объект. Null и отсутствующая колонка дают
  пустой объект. Вложенные объекты и массивы поддерживаются. Ключи — строки,
  числа — конечные. Даты заранее переводятся в ISO 8601, NumPy-скаляры —
  в обычные JSON-скаляры.
- `embedding_size`: необязательная входная колонка. Если присутствует,
  каждое значение — целое, равное `embedding_dim`.

Канонический writer сохраняет `id: string`, `name: string`,
`embedding: fixed_size_list<float32>[D]`, `metadata: string` с JSON-объектом.
Reader принимает также Arrow struct для metadata. Дополнительные колонки
не становятся metadata автоматически; загрузчик предупреждает об их пропуске.
Используйте адаптер или заранее соберите объект metadata.

Нулевые векторы разрешены, их ID присутствуют в отчёте. Для cosine выдаётся
предупреждение: косинус нулевого вектора не определён. Пустой набор запрещён.
Диагностика содержит первые 20 ошибок и общий счётчик; номера строк начинаются с 1.
Отчёт перечисляет типы верхнеуровневых metadata-полей, различая отсутствие и null.

## Python API

```python
from pathlib import Path
import numpy as np
from embedding_visualizer.data import DatasetManifest, load_dataset, write_dataset

manifest = DatasetManifest("Example", "model:layer", "1", 3)
write_dataset(
    Path("data/example-api"),
    ids=["a", "b"],
    embeddings=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
    manifest=manifest,
    names=["First", "Second"],
    metadata=[{"group": "x"}, {"group": "y", "count": 7}],
)
dataset = load_dataset(Path("data/example-api"))
print(dataset.embeddings.shape)
print(dataset.report.metadata_fields)
```

`Dataset` содержит IDs, имена, float32-матрицу, metadata, manifest и report.
Все массивы имеют общий порядок строк.

Writer сначала проверяет и готовит файлы, затем устанавливает их в выходную
папку. Замена существующих файлов требует `overwrite=True`; есть откат при
ошибке записи. Это не транзакция для конкурентных читателей/писателей и не
защита от остановки процесса между заменой двух файлов.
