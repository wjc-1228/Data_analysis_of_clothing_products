import json
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm  # 可选，显示进度条

def jsonl_to_parquet(jsonl_path, parquet_path, chunk_size=10000):
    """
    将 JSON Lines 文件转换为 Parquet 格式（分块处理，内存友好）
    
    Parameters:
        jsonl_path (str): 输入 JSONL 文件路径
        parquet_path (str): 输出 Parquet 文件路径
        chunk_size (int): 每批次处理的行数（根据内存调整）
    """
    writer = None
    total_lines = 0  # 仅用于进度条，可省略

    # 先计算总行数（用于进度条），如果文件太大可跳过
    # with open(jsonl_path, 'r', encoding='utf-8') as f:
    #     total_lines = sum(1 for _ in f)

    with open(jsonl_path, 'r', encoding='utf-8') as infile:
        # 使用 tqdm 显示进度（可选）
        # pbar = tqdm(total=total_lines, desc="转换进度")
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
                # pbar.update(chunk_size)

        # 处理剩余行
        if batch:
            _write_batch(batch, writer, parquet_path, is_first=(writer is None))
            # pbar.update(len(batch))

    # if pbar:
    #     pbar.close()
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

# 使用示例
input_file = 'D:/workspace/Data_analysis_of_clothing_products/meta_Clothing_Shoes_and_Jewelry.jsonl'
output_file = 'D:/workspace/Data_analysis_of_clothing_products/meta.parquet'
jsonl_to_parquet(input_file, output_file, chunk_size=50000)  # 可根据内存调整