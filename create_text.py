#%%
import os
import re

# Allowed characters
allowed_chars = set([
    '\t', '\n', ' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '[', ']', '^', '_',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
    'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '{', '|', '}', '-', '"'
])

# Character normalization mappings
normalize_map = {
    '‘': '"', '’': '"',
    '“': '"', '”': '"', '′': "'",
    '–': '-', '—': '-',
}

def normalize_and_filter_characters(text):
    # Normalize special quotes and dashes
    for orig, repl in normalize_map.items():
        text = text.replace(orig, repl)
    # Remove characters not in allowed list
    return ''.join(c for c in text if c in allowed_chars)

def clean_text(text):
    lines = text.splitlines()

    cleaned = []

    roman_numeral_regex = re.compile(
        r'^\s*(?:M{0,4}(CM|CD|D?C{0,3})?(XC|XL|L?X{0,3})?(IX|IV|V?I{0,3})?)\.?\s*$',
        re.IGNORECASE
    )
    number_line_regex = re.compile(r'^\s*\d+(\.\d+)?\.?\s*$')
    special_chars_only_regex = re.compile(r'^[^a-zA-Z0-9]+$')

    # Step 1: Remove junk lines
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
        elif (
            roman_numeral_regex.match(stripped) or
            number_line_regex.match(stripped) or
            special_chars_only_regex.match(stripped)
        ):
            continue
        else:
            cleaned.append(line)

    # Step 2: Remove lines starting with "NOTE", "note", or "["
    filtered_lines = [line for line in cleaned if not re.match(r'^(NOTE|note|\[)', line.strip())]

    # Step 3: Remove inline patterns like [6], [Pg 6], [pg 003]
    cleaned_lines = [re.sub(r'\[\s*(?:pg|Pg)?\s*\d+\s*\]', '', line) for line in filtered_lines]

    # Step 4: Normalize and remove unwanted characters
    normalized_lines = [normalize_and_filter_characters(line) for line in cleaned_lines]

    # Step 5: Remove consecutive empty lines (keep only one)
    final_lines = []
    previous_blank = False
    for line in normalized_lines:
        if line.strip() == "":
            if not previous_blank:
                final_lines.append("")
            previous_blank = True
        else:
            final_lines.append(line)
            previous_blank = False

    return "\n".join(final_lines)

def concatenate_texts_from_directory(dir_path, output_filename):
    output_path = os.path.join('', output_filename)
    all_text = ""
    for filename in os.listdir(dir_path):
        if filename.endswith(".txt") and filename != output_filename:
            file_path = os.path.join(dir_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
                cleaned_text = clean_text(raw_text)
                all_text += cleaned_text + "\n"
    with open(output_path, "w", encoding="utf-8") as out_file:
        out_file.write(all_text)
    print(f"\nCombined and cleaned text saved to: {output_path}")

#%%
concatenate_texts_from_directory("./philosophy", "philosophy.txt")