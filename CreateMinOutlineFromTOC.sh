#!/bin/bash

source /home/jesse/Dropbox/MCGLAW/config_files/scripts/misc/system_wide.sh
source /home/jesse/Dropbox/MCGLAW/config_files/scripts/misc/model_functions.sh

# Model to use for one-sentence minute descriptions
MODEL="fireworks_deepseek_3_2"

# Pull in selection-derived variables (e.g., $file_path, $dir_path, $case_number, $case_name)
create_arguments_infer_variables

# Paths and scratch space
TEMP_ENV="/dev/shm/CreateMinOutline"
TOC_RAW="${TEMP_ENV}/toc.txt"
MINUTES_TEXT="${TEMP_ENV}/minutes_outline.txt"
MINUTES_PDF="${TEMP_ENV}/minutes_outline.pdf"

# Create/refresh temp working area and bring toc.txt into it
create_temp_environment "$TEMP_ENV"
cp "$file_path" "$TOC_RAW"

# Extract the MINUTES lines from the toc (trim leading tabs/spaces)
mapfile -t MINUTE_ENTRIES < <(
  awk '
    function ltrim(s) { sub(/^[ \t]+/, "", s); return s }
    /^MINUTES[[:space:]]/ { in_minutes=1; next }
    in_minutes && /^[^ \t]/ { in_minutes=0 }
    in_minutes && /^[ \t]+/ { print ltrim($0) }
  ' "$TOC_RAW"
)

# Start the outline
{
echo "Minutes Summary"
  echo "$case_number $case_name"
  echo
} > "$MINUTES_TEXT"

# Build the outline with model-generated one-liners for each minutes entry
for entry in "${MINUTE_ENTRIES[@]}"; do
  # Split into date text and page id (last token)
  page_id=$(printf '%s\n' "$entry" | awk '{print $NF}')
  date_text=$(printf '%s\n' "$entry" | sed 's/[[:space:]]*[0-9][0-9]*$//' | sed 's/[[:space:]]*$//')

  # Preserve width when incrementing (e.g., 0407 -> 0408)
  page_width=${#page_id}
  page_id_next=$(printf "%0${page_width}d" $((10#$page_id + 1)))

  page_file="${dir_path}/${page_id}.txt"
  page_file_next="${dir_path}/${page_id_next}.txt"

  # Date line; blank line below ensures the summary renders on its own line in PDF
  echo "**${date_text}**" >> "$MINUTES_TEXT"
  echo >> "$MINUTES_TEXT"

  if [[ -f "$page_file" || -f "$page_file_next" ]]; then
    prompt_file=$(mktemp)
    {
      echo -n "I will provide you with the first two pages of a minute order. Based on this information, state the name of the hearing, whether the hearing was reported, whether one or both parents were present, and what the juvenile court ordered. The description of what the juvenile court ordered must be brief and concise. Only state that a parent is present if the minute order indicates that the parent is present on the first page of the minute order. If only a parent's attorney is listed, assume that the parent is not present. Do not insert any line breaks. Here are three examples of the proper format:

Detention Hearing. Reported. No parent appeared. The juvenile court ordered the children temporarily removed from the parents.

Receipt of Report Hearing. Not Reported. No parent appeared. The juvenile court received the section 361.66 report into evidence.

Permanent Plan Review Hearing. Reported. Only mother appeared. The juvenile court received the social worker reports into evidence and heard testimony from mother. The juvenile court terminated parental rights.

Okay, here is the minute orders: "
      [[ -f "$page_file" ]] && cat "$page_file"
      [[ -f "$page_file_next" ]] && cat "$page_file_next"
    } > "$prompt_file"

    RESPONSE=$($MODEL "$prompt_file")
    RESPONSE=$(echo "$RESPONSE" | tr -s ' ')  # Collapse any accidental double spaces from the model
    rm -f "$prompt_file"
    echo "$RESPONSE" >> "$MINUTES_TEXT"
  else
    echo "Missing source file(s): ${page_id}.txt and/or ${page_id_next}.txt" >> "$MINUTES_TEXT"
  fi

  echo >> "$MINUTES_TEXT"
done

# Create PDF version of the outline
cd "$TEMP_ENV"
pandoc "$MINUTES_TEXT" \
  -o "$MINUTES_PDF" \
  --pdf-engine=xelatex \
  -V mainfont="Century Schoolbook" \
  -V fontsize=11pt \
  -V geometry:margin=30mm

# Copy PDF to 0_record (parent of text_record); keep working copies in shm
case_root=$(dirname "$dir_path")
if [[ -d "$case_root" ]]; then
  cp "$MINUTES_PDF" "$case_root/minutes_outline.pdf"
fi

exit
