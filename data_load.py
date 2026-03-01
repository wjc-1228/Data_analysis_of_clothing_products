import json
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm  

def jsonl_to_parquet(jsonl_path, parquet_path, chunk_size=10000):
    writer = None
    total_lines = 0  

    with open(jsonl_path, 'r', encoding='utf-8') as infile:

        batch = []
        for i, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"跳过无效行 {i}: {e}")
                continue
            batch.append(obj)

            # 达到块大小，写入 Parquet
            if len(batch) >= chunk_size:
                _write_batch(batch, writer, parquet_path, is_first=(writer is None))
                batch = []

        # 处理剩余行
        if batch:
            _write_batch(batch, writer, parquet_path, is_first=(writer is None))


    if writer:
        writer.close()
    print(f"转换完成！输出文件：{parquet_path}")

def _write_batch(batch, writer, parquet_path, is_first):
    """将一批 Python 对象写入 Parquet"""
    # 将列表转换为 PyArrow Table
    table = pa.Table.from_pylist(batch)
    
    if is_first:
        # 第一个批次：创建 ParquetWriter 并写入
        writer = pq.ParquetWriter(parquet_path, table.schema)
        writer.write_table(table)
    else:
        # 后续批次：直接写入已存在的 writer
        writer.write_table(table)
    
    return writer


input_file = 'D:/workspace/Data_analysis_of_clothing_products/meta_Clothing_Shoes_and_Jewelry.jsonl'
output_file = 'D:/workspace/Data_analysis_of_clothing_products/meta.parquet'

jsonl_to_parquet(input_file, output_file, chunk_size=100000)  
