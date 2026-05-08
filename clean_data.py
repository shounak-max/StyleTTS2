import os

def clean_file(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
    
    print(f"Cleaning {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            # Ensure only 3 pipes, normalize spacing
            parts = [p.strip() for p in stripped.split('|')]
            clean_lines.append('|'.join(parts))
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(clean_lines) + '\n')
    print(f"Done. Cleaned {len(lines)} lines.")

if __name__ == "__main__":
    clean_file("Data/bn_train_list.txt")
    clean_file("Data/bn_val_list.txt")
    clean_file("Data/bn_OOD_texts.txt")
