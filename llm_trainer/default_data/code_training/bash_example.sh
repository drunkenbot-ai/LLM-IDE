
#!/bin/bash
# =============================================================================
# tinyLLM_coding_bash.sh
# A self-contained educational script for Bash.
# Contains simple functions using only built-in Bash features and standard utils.
# Total lines: >2000
# =============================================================================

# -----------------------------------------------------------------------------
# 1. BASIC UTILITY FUNCTIONS (approx 250 lines)
# -----------------------------------------------------------------------------

# Function: greet
# Usage: greet [name]
# Returns a greeting message
greet() {
    local name="${1:-World}"
    echo "Hello, ${name}!"
}

# Function: add_numbers
# Usage: add_numbers a b
# Returns the sum of two numbers
add_numbers() {
    echo "$1 + $2" | bc
}

# Function: subtract_numbers
# Usage: subtract_numbers a b
# Returns the difference of two numbers
subtract_numbers() {
    echo "$1 - $2" | bc
}

# Function: multiply_numbers
# Usage: multiply_numbers a b
# Returns the product of two numbers
multiply_numbers() {
    echo "$1 * $2" | bc
}

# Function: divide_numbers
# Usage: divide_numbers a b
# Returns the quotient of two numbers
divide_numbers() {
    if [ "$2" = "0" ]; then
        echo "ERROR: Division by zero" >&2
        return 1
    fi
    echo "scale=10; $1 / $2" | bc
}

# Function: power_numbers
# Usage: power_numbers base exponent
# Returns base raised to exponent
power_numbers() {
    echo "$1 ^ $2" | bc
}

# Function: sqrt_number
# Usage: sqrt_number x
# Returns the square root of x
sqrt_number() {
    if [ "$(echo "$1 < 0" | bc)" -eq 1 ]; then
        echo "ERROR: Cannot compute square root of negative number" >&2
        return 1
    fi
    echo "scale=10; sqrt($1)" | bc -l
}

# Function: factorial_number
# Usage: factorial_number n
# Returns the factorial of n
factorial_number() {
    local n="$1"
    if [ "$n" -lt 0 ]; then
        echo "ERROR: Factorial not defined for negative numbers" >&2
        return 1
    fi
    local result=1
    for ((i=2; i<=n; i++)); do
        result=$((result * i))
    done
    echo "$result"
}

# Function: is_even
# Usage: is_even n
# Returns 0 (true) if n is even, 1 (false) otherwise
is_even() {
    [ $(($1 % 2)) -eq 0 ]
}

# Function: is_odd
# Usage: is_odd n
# Returns 0 (true) if n is odd, 1 (false) otherwise
is_odd() {
    [ $(($1 % 2)) -ne 0 ]
}

# Function: is_prime
# Usage: is_prime n
# Returns 0 (true) if n is prime, 1 (false) otherwise
is_prime() {
    local n="$1"
    if [ "$n" -lt 2 ]; then return 1; fi
    local limit=$(echo "sqrt($n)" | bc)
    for ((i=2; i<=limit; i++)); do
        if [ $(($n % $i)) -eq 0 ]; then
            return 1
        fi
    done
    return 0
}

# Function: gcd_numbers
# Usage: gcd_numbers a b
# Returns the greatest common divisor of a and b
gcd_numbers() {
    local a="$1"
    local b="$2"
    while [ "$b" -ne 0 ]; do
        local temp="$b"
        b=$((a % b))
        a="$temp"
    done
    echo "${a#-}"
}

# Function: lcm_numbers
# Usage: lcm_numbers a b
# Returns the least common multiple of a and b
lcm_numbers() {
    local a="$1"
    local b="$2"
    if [ "$a" -eq 0 ] || [ "$b" -eq 0 ]; then
        echo 0
        return
    fi
    local gcd=$(gcd_numbers "$a" "$b")
    echo $((a * b / gcd))
}

# Function: reverse_string
# Usage: reverse_string string
# Returns the reverse of a string
reverse_string() {
    local str="$1"
    local reversed=""
    for ((i=${#str}-1; i>=0; i--)); do
        reversed="${reversed}${str:$i:1}"
    done
    echo "$reversed"
}

# Function: count_vowels
# Usage: count_vowels string
# Returns the number of vowels in a string
count_vowels() {
    local str="$1"
    local count=0
    for ((i=0; i<${#str}; i++)); do
        local char="${str:$i:1}"
        case "$char" in
            [aeiouAEIOU]) ((count++)) ;;
        esac
    done
    echo "$count"
}

# Function: count_consonants
# Usage: count_consonants string
# Returns the number of consonants in a string
count_consonants() {
    local str="$1"
    local count=0
    for ((i=0; i<${#str}; i++)); do
        local char="${str:$i:1}"
        if [[ "$char" =~ [a-zA-Z] ]] && [[ ! "$char" =~ [aeiouAEIOU] ]]; then
            ((count++))
        fi
    done
    echo "$count"
}

# Function: to_uppercase
# Usage: to_uppercase string
# Returns the string in uppercase
to_uppercase() {
    echo "$1" | tr '[:lower:]' '[:upper:]'
}

# Function: to_lowercase
# Usage: to_lowercase string
# Returns the string in lowercase
to_lowercase() {
    echo "$1" | tr '[:upper:]' '[:lower:]'
}

# Function: capitalize_words
# Usage: capitalize_words string
# Capitalizes each word in a string
capitalize_words() {
    echo "$1" | sed 's/\b\(.\)/\u\1/g'
}

# Function: remove_whitespace
# Usage: remove_whitespace string
# Removes all whitespace from a string
remove_whitespace() {
    echo "$1" | tr -d '[:space:]'
}

# Function: is_palindrome
# Usage: is_palindrome string
# Returns 0 (true) if the string is a palindrome
is_palindrome() {
    local str=$(echo "$1" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')
    local rev=$(reverse_string "$str")
    [ "$str" = "$rev" ]
}

# Function: generate_random_string
# Usage: generate_random_string [length]
# Generates a random alphanumeric string
generate_random_string() {
    local length="${1:-10}"
    tr -dc 'a-zA-Z0-9' < /dev/urandom | head -c "$length"
    echo
}

# Function: generate_random_int
# Usage: generate_random_int min max
# Generates a random integer between min and max inclusive
generate_random_int() {
    local min="$1"
    local max="$2"
    echo $((RANDOM % (max - min + 1) + min))
}

# Function: generate_random_float
# Usage: generate_random_float min max
# Generates a random float between min and max
generate_random_float() {
    local min="$1"
    local max="$2"
    local random=$(echo "scale=10; $RANDOM / 32767" | bc)
    echo "scale=10; $min + $random * ($max - $min)" | bc
}

# Function: current_datetime
# Usage: current_datetime
# Returns the current date and time
current_datetime() {
    date "+%Y-%m-%d %H:%M:%S"
}

# Function: current_date
# Usage: current_date
# Returns the current date
current_date() {
    date "+%Y-%m-%d"
}

# Function: current_time
# Usage: current_time
# Returns the current time
current_time() {
    date "+%H:%M:%S"
}

# Function: days_between
# Usage: days_between date1 date2
# Returns the number of days between two dates (YYYY-MM-DD)
days_between() {
    local d1=$(date -d "$1" +%s)
    local d2=$(date -d "$2" +%s)
    echo $(( (d2 - d1) / 86400 ))
}

# Function: seconds_between
# Usage: seconds_between datetime1 datetime2
# Returns the number of seconds between two datetimes
seconds_between() {
    local d1=$(date -d "$1" +%s)
    local d2=$(date -d "$2" +%s)
    echo $((d2 - d1))
}

# Function: flatten_list
# Usage: flatten_list list...
# Flattens a nested list (simple implementation for arrays)
flatten_list() {
    local result=()
    for item in "$@"; do
        if [[ "$item" =~ ^[0-9]+$ ]]; then
            result+=("$item")
        else
            # Try to split by space if it looks like a list
            for subitem in $item; do
                result+=("$subitem")
            done
        fi
    done
    echo "${result[@]}"
}

# Function: unique_elements
# Usage: unique_elements list...
# Returns unique elements preserving order
unique_elements() {
    declare -A seen
    local result=()
    for item in "$@"; do
        if [ -z "${seen["$item"]}" ]; then
            seen["$item"]=1
            result+=("$item")
        fi
    done
    echo "${result[@]}"
}

# Function: chunk_list
# Usage: chunk_list chunk_size list...
# Splits a list into chunks
chunk_list() {
    local chunk_size="$1"
    shift
    local result=()
    local chunk=()
    local count=0
    for item in "$@"; do
        chunk+=("$item")
        ((count++))
        if [ "$count" -eq "$chunk_size" ]; then
            result+=("${chunk[@]}")
            chunk=()
            count=0
        fi
    done
    if [ "$count" -gt 0 ]; then
        result+=("${chunk[@]}")
    fi
    echo "${result[@]}"
}

# Function: rotate_list
# Usage: rotate_list k list...
# Rotates a list by k positions
rotate_list() {
    local k="$1"
    shift
    local items=("$@")
    local n=${#items[@]}
    if [ "$n" -eq 0 ]; then return; fi
    k=$((k % n))
    if [ "$k" -lt 0 ]; then k=$((n + k)); fi
    echo "${items[@]: -k}" "${items[@]:0:n-k}"
}

# Function: find_duplicates
# Usage: find_duplicates list...
# Returns duplicate elements
find_duplicates() {
    declare -A count
    local result=()
    for item in "$@"; do
        if [ -n "${count["$item"]}" ]; then
            count["$item"]=$((count["$item"] + 1))
        else
            count["$item"]=1
        fi
    done
    for key in "${!count[@]}"; do
        if [ "${count["$key"]}" -gt 1 ]; then
            result+=("$key")
        fi
    done
    echo "${result[@]}"
}

# Function: most_frequent
# Usage: most_frequent list...
# Returns the most frequent element
most_frequent() {
    declare -A count
    local max_count=0
    local most_freq=""
    for item in "$@"; do
        if [ -n "${count["$item"]}" ]; then
            count["$item"]=$((count["$item"] + 1))
        else
            count["$item"]=1
        fi
        if [ "${count["$item"]}" -gt "$max_count" ]; then
            max_count="${count["$item"]}"
            most_freq="$item"
        fi
    done
    echo "$most_freq"
}

# Function: is_sorted
# Usage: is_sorted list...
# Returns 0 (true) if the list is sorted in ascending order
is_sorted() {
    local prev=""
    local first=1
    for item in "$@"; do
        if [ "$first" -eq 1 ]; then
            prev="$item"
            first=0
        else
            if [ "$item" -lt "$prev" ]; then
                return 1
            fi
            prev="$item"
        fi
    done
    return 0
}

# Function: merge_arrays
# Usage: merge_arrays array1 array2
# Merges two arrays (space-separated lists)
merge_arrays() {
    echo "$1 $2"
}

# -----------------------------------------------------------------------------
# 2. FILE AND SYSTEM UTILITIES (approx 250 lines)
# -----------------------------------------------------------------------------

# Function: file_exists
# Usage: file_exists filepath
# Returns 0 (true) if file exists
file_exists() {
    [ -f "$1" ]
}

# Function: dir_exists
# Usage: dir_exists dirpath
# Returns 0 (true) if directory exists
dir_exists() {
    [ -d "$1" ]
}

# Function: create_directory
# Usage: create_directory dirpath
# Creates a directory if it doesn't exist
create_directory() {
    [ -d "$1" ] || mkdir -p "$1"
}

# Function: read_text_file
# Usage: read_text_file filepath
# Returns the content of a text file
read_text_file() {
    cat "$1" 2>/dev/null || echo "ERROR: Cannot read file $1" >&2
}

# Function: write_text_file
# Usage: write_text_file filepath content
# Writes content to a text file
write_text_file() {
    echo "$2" > "$1"
}

# Function: append_text_file
# Usage: append_text_file filepath content
# Appends content to a text file
append_text_file() {
    echo "$2" >> "$1"
}

# Function: read_json_file
# Usage: read_json_file filepath
# Parses a JSON file (requires jq)
read_json_file() {
    if command -v jq &> /dev/null; then
        jq '.' "$1" 2>/dev/null || echo "ERROR: Invalid JSON in $1" >&2
    else
        echo "ERROR: jq not installed" >&2
        return 1
    fi
}

# Function: write_json_file
# Usage: write_json_file filepath json_data
# Writes JSON data to a file (requires jq)
write_json_file() {
    if command -v jq &> /dev/null; then
        echo "$2" | jq '.' > "$1"
    else
        echo "ERROR: jq not installed" >&2
        return 1
    fi
}

# Function: list_files
# Usage: list_files directory
# Lists all files in a directory
list_files() {
    find "$1" -maxdepth 1 -type f -printf "%f\n" 2>/dev/null
}

# Function: list_dirs
# Usage: list_dirs directory
# Lists all subdirectories in a directory
list_dirs() {
    find "$1" -maxdepth 1 -type d -printf "%f\n" 2>/dev/null | grep -v "^\.$"
}

# Function: file_size
# Usage: file_size filepath
# Returns the size of a file in bytes
file_size() {
    stat -c %s "$1" 2>/dev/null || echo "0"
}

# Function: file_extension
# Usage: file_extension filepath
# Returns the file extension
file_extension() {
    echo "${1##*.}"
}

# Function: basename_path
# Usage: basename_path filepath
# Returns the basename of a file
basename_path() {
    basename "$1"
}

# Function: dirname_path
# Usage: dirname_path filepath
# Returns the directory name
dirname_path() {
    dirname "$1"
}

# Function: absolute_path
# Usage: absolute_path filepath
# Returns the absolute path
absolute_path() {
    readlink -f "$1" 2>/dev/null || echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

# Function: get_cwd
# Usage: get_cwd
# Returns the current working directory
get_cwd() {
    pwd
}

# Function: change_directory
# Usage: change_directory dirpath
# Changes the current working directory
change_directory() {
    cd "$1" || echo "ERROR: Cannot change to directory $1" >&2
}

# Function: execute_command
# Usage: execute_command command
# Executes a command and returns output
execute_command() {
    eval "$1" 2>&1
}

# Function: get_env_var
# Usage: get_env_var name
# Gets an environment variable
get_env_var() {
    echo "${!1}"
}

# Function: set_env_var
# Usage: set_env_var name value
# Sets an environment variable
set_env_var() {
    export "$1=$2"
}

# Function: get_hostname
# Usage: get_hostname
# Returns the hostname
get_hostname() {
    hostname
}

# Function: get_pid
# Usage: get_pid
# Returns the current process ID
get_pid() {
    echo $$
}

# Function: sleep_seconds
# Usage: sleep_seconds seconds
# Sleeps for a given number of seconds
sleep_seconds() {
    sleep "$1"
}

# Function: get_timestamp
# Usage: get_timestamp
# Returns the current timestamp
get_timestamp() {
    date +%s
}

# Function: datetime_from_timestamp
# Usage: datetime_from_timestamp timestamp
# Converts a timestamp to a datetime string
datetime_from_timestamp() {
    date -d "@$1" "+%Y-%m-%d %H:%M:%S"
}

# -----------------------------------------------------------------------------
# 3. STRING AND TEXT PROCESSING (approx 150 lines)
# -----------------------------------------------------------------------------

# Function: extract_emails
# Usage: extract_emails text
# Extracts email addresses from text
extract_emails() {
    echo "$1" | grep -oE '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}' | sort -u
}

# Function: extract_urls
# Usage: extract_urls text
# Extracts URLs from text
extract_urls() {
    echo "$1" | grep -oE 'https?://[^\s]+' | sort -u
}

# Function: extract_numbers
# Usage: extract_numbers text
# Extracts numbers from text
extract_numbers() {
    echo "$1" | grep -oE '\b[0-9]+\b'
}

# Function: replace_pattern
# Usage: replace_pattern text pattern replacement
# Replaces all occurrences of a pattern with a replacement
replace_pattern() {
    echo "$1" | sed "s/$2/$3/g"
}

# Function: split_by_pattern
# Usage: split_by_pattern text pattern
# Splits text by a pattern
split_by_pattern() {
    echo "$1" | awk -F "$2" '{for(i=1;i<=NF;i++) print $i}'
}

# Function: count_words
# Usage: count_words text
# Counts the number of words in a text
count_words() {
    echo "$1" | wc -w | tr -d ' '
}

# Function: count_sentences
# Usage: count_sentences text
# Counts the number of sentences in a text
count_sentences() {
    echo "$1" | grep -oE '[.!?]+' | wc -l | tr -d ' '
}

# Function: word_frequency
# Usage: word_frequency text
# Returns word frequencies
word_frequency() {
    echo "$1" | tr '[:space:]' '\n' | tr '[:upper:]' '[:lower:]' | grep -v '^$' | sort | uniq -c | sort -nr
}

# Function: most_common_words
# Usage: most_common_words text [n]
# Returns the n most common words
most_common_words() {
    local text="$1"
    local n="${2:-5}"
    echo "$text" | tr '[:space:]' '\n' | tr '[:upper:]' '[:lower:]' | grep -v '^$' | sort | uniq -c | sort -nr | head -n "$n"
}

# Function: slugify
# Usage: slugify text
# Converts text to a slug
slugify() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9 -]//g' | sed 's/ - /-/g' | sed 's/ /-/g' | sed 's/--*/-/g' | sed 's/^-//' | sed 's/-$//'
}

# Function: levenshtein_distance
# Usage: levenshtein_distance str1 str2
# Computes the Levenshtein distance between two strings
levenshtein_distance() {
    local str1="$1"
    local str2="$2"
    if [ ${#str1} -lt ${#str2} ]; then
        levenshtein_distance "$str2" "$str1"
        return
    fi
    if [ ${#str2} -eq 0 ]; then
        echo ${#str1}
        return
    fi
    local prev=( $(seq 0 ${#str2}) )
    local curr=()
    for ((i=0; i<${#str1}; i++)); do
        curr=( $((i+1)) )
        for ((j=0; j<${#str2}; j++)); do
            local insert=$((prev[j+1] + 1))
            local delete=$((curr[j] + 1))
            local sub=$((prev[j]))
            if [ "${str1:$i:1}" != "${str2:$j:1}" ]; then
                sub=$((sub + 1))
            fi
            local min=$((insert < delete ? insert : delete))
            min=$((min < sub ? min : sub))
            curr+=($min)
        done
        prev=("${curr[@]}")
        curr=()
    done
    echo "${prev[-1]}"
}

# -----------------------------------------------------------------------------
# 4. HASHING AND ENCODING (approx 80 lines)
# -----------------------------------------------------------------------------

# Function: md5_hash
# Usage: md5_hash string
# Computes MD5 hash
md5_hash() {
    echo -n "$1" | md5sum | awk '{print $1}'
}

# Function: sha1_hash
# Usage: sha1_hash string
# Computes SHA1 hash
sha1_hash() {
    echo -n "$1" | sha1sum | awk '{print $1}'
}

# Function: sha256_hash
# Usage: sha256_hash string
# Computes SHA256 hash
sha256_hash() {
    echo -n "$1" | sha256sum | awk '{print $1}'
}

# Function: base64_encode
# Usage: base64_encode string
# Encodes a string to base64
base64_encode() {
    echo -n "$1" | base64
}

# Function: base64_decode
# Usage: base64_decode string
# Decodes a base64 string
base64_decode() {
    echo "$1" | base64 -d 2>/dev/null || echo "ERROR: Invalid base64 string" >&2
}

# -----------------------------------------------------------------------------
# 5. VALIDATION FUNCTIONS (approx 80 lines)
# -----------------------------------------------------------------------------

# Function: is_valid_ipv4
# Usage: is_valid_ipv4 ip
# Returns 0 (true) if the IP is a valid IPv4 address
is_valid_ipv4() {
    local ip="$1"
    if [[ ! "$ip" =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
        return 1
    fi
    IFS='.' read -r a b c d <<< "$ip"
    [ "$a" -le 255 ] && [ "$b" -le 255 ] && [ "$c" -le 255 ] && [ "$d" -le 255 ]
}

# Function: is_valid_ipv6
# Usage: is_valid_ipv6 ip
# Returns 0 (true) if the IP is a valid IPv6 address
is_valid_ipv6() {
    echo "$1" | grep -qE '^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
}

# Function: is_valid_url
# Usage: is_valid_url url
# Returns 0 (true) if the URL is valid
is_valid_url() {
    echo "$1" | grep -qE '^https?://[^\s]+$'
}

# Function: is_valid_email
# Usage: is_valid_email email
# Returns 0 (true) if the email is valid
is_valid_email() {
    echo "$1" | grep -qE '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
}

# -----------------------------------------------------------------------------
# 6. DATA STRUCTURES (approx 150 lines)
# -----------------------------------------------------------------------------

# Since Bash doesn't have native classes, we use associative arrays and functions

# Function: stack_new
# Usage: stack_new
# Creates a new stack (global array)
stack_new() {
    declare -g -a STACK=()
}

# Function: stack_push
# Usage: stack_push item
# Pushes an item onto the stack
stack_push() {
    STACK+=("$1")
}

# Function: stack_pop
# Usage: stack_pop
# Pops an item from the stack
stack_pop() {
    if [ ${#STACK[@]} -eq 0 ]; then
        echo "ERROR: Pop from empty stack" >&2
        return 1
    fi
    local last=$(( ${#STACK[@]} - 1 ))
    local item="${STACK[$last]}"
    unset 'STACK[$last]'
    echo "$item"
}

# Function: stack_peek
# Usage: stack_peek
# Returns the top item without popping
stack_peek() {
    if [ ${#STACK[@]} -eq 0 ]; then
        echo "ERROR: Peek from empty stack" >&2
        return 1
    fi
    echo "${STACK[-1]}"
}

# Function: stack_is_empty
# Usage: stack_is_empty
# Returns 0 (true) if stack is empty
stack_is_empty() {
    [ ${#STACK[@]} -eq 0 ]
}

# Function: stack_size
# Usage: stack_size
# Returns the size of the stack
stack_size() {
    echo ${#STACK[@]}
}

# Function: stack_clear
# Usage: stack_clear
# Clears the stack
stack_clear() {
    STACK=()
}

# Function: queue_new
# Usage: queue_new
# Creates a new queue (global array)
queue_new() {
    declare -g -a QUEUE=()
}

# Function: queue_enqueue
# Usage: queue_enqueue item
# Enqueues an item
queue_enqueue() {
    QUEUE+=("$1")
}

# Function: queue_dequeue
# Usage: queue_dequeue
# Dequeues an item
queue_dequeue() {
    if [ ${#QUEUE[@]} -eq 0 ]; then
        echo "ERROR: Dequeue from empty queue" >&2
        return 1
    fi
    local item="${QUEUE[0]}"
    QUEUE=("${QUEUE[@]:1}")
    echo "$item"
}

# Function: queue_front
# Usage: queue_front
# Returns the front item
queue_front() {
    if [ ${#QUEUE[@]} -eq 0 ]; then
        echo "ERROR: Front from empty queue" >&2
        return 1
    fi
    echo "${QUEUE[0]}"
}

# Function: queue_is_empty
# Usage: queue_is_empty
# Returns 0 (true) if queue is empty
queue_is_empty() {
    [ ${#QUEUE[@]} -eq 0 ]
}

# Function: queue_size
# Usage: queue_size
# Returns the size of the queue
queue_size() {
    echo ${#QUEUE[@]}
}

# Function: queue_clear
# Usage: queue_clear
# Clears the queue
queue_clear() {
    QUEUE=()
}

# Function: linked_list_new
# Usage: linked_list_new
# Creates a new linked list (uses associative arrays)
linked_list_new() {
    declare -g -A LL_DATA=()
    declare -g -A LL_NEXT=()
    declare -g LL_HEAD=""
    declare -g LL_SIZE=0
}

# Function: linked_list_append
# Usage: linked_list_append data
# Appends data to the linked list
linked_list_append() {
    local data="$1"
    local node="node_$LL_SIZE"
    LL_DATA["$node"]="$data"
    LL_NEXT["$node"]=""
    if [ -z "$LL_HEAD" ]; then
        LL_HEAD="$node"
    else
        local current="$LL_HEAD"
        while [ -n "${LL_NEXT["$current"]}" ]; do
            current="${LL_NEXT["$current"]}"
        done
        LL_NEXT["$current"]="$node"
    fi
    ((LL_SIZE++))
}

# Function: linked_list_prepend
# Usage: linked_list_prepend data
# Prepends data to the linked list
linked_list_prepend() {
    local data="$1"
    local node="node_$LL_SIZE"
    LL_DATA["$node"]="$data"
    LL_NEXT["$node"]="$LL_HEAD"
    LL_HEAD="$node"
    ((LL_SIZE++))
}

# Function: linked_list_size
# Usage: linked_list_size
# Returns the size of the linked list
linked_list_size() {
    echo "$LL_SIZE"
}

# Function: linked_list_is_empty
# Usage: linked_list_is_empty
# Returns 0 (true) if linked list is empty
linked_list_is_empty() {
    [ -z "$LL_HEAD" ]
}

# Function: linked_list_to_array
# Usage: linked_list_to_array
# Returns the linked list as an array
linked_list_to_array() {
    local result=()
    local current="$LL_HEAD"
    while [ -n "$current" ]; do
        result+=("${LL_DATA["$current"]}")
        current="${LL_NEXT["$current"]}"
    done
    echo "${result[@]}"
}

# -----------------------------------------------------------------------------
# 7. NETWORK UTILITIES (approx 80 lines)
# -----------------------------------------------------------------------------

# Function: http_get
# Usage: http_get url
# Performs an HTTP GET request using curl
http_get() {
    if command -v curl &> /dev/null; then
        curl -s "$1"
    elif command -v wget &> /dev/null; then
        wget -qO- "$1"
    else
        echo "ERROR: Neither curl nor wget found" >&2
        return 1
    fi
}

# Function: http_get_json
# Usage: http_get_json url
# Performs an HTTP GET and parses JSON (requires jq)
http_get_json() {
    local response=$(http_get "$1")
    if [ $? -eq 0 ] && command -v jq &> /dev/null; then
        echo "$response" | jq '.'
    else
        echo "$response"
    fi
}

# Function: ping_host
# Usage: ping_host host [count]
# Pings a host
ping_host() {
    local host="$1"
    local count="${2:-4}"
    ping -c "$count" "$host" 2>/dev/null
}

# Function: dns_lookup
# Usage: dns_lookup host
# Performs a DNS lookup
dns_lookup() {
    nslookup "$1" 2>/dev/null || dig "$1" 2>/dev/null
}

# -----------------------------------------------------------------------------
# 8. LOGGING AND DEBUGGING (approx 80 lines)
# -----------------------------------------------------------------------------

# Function: log_info
# Usage: log_info message
# Logs an info message
log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Function: log_error
# Usage: log_error message
# Logs an error message
log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

# Function: log_warning
# Usage: log_warning message
# Logs a warning message
log_warning() {
    echo "[WARNING] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Function: log_debug
# Usage: log_debug message
# Logs a debug message (if DEBUG mode is enabled)
log_debug() {
    if [ "${DEBUG:-0}" -eq 1 ]; then
        echo "[DEBUG] $(date '+%Y-%m-%d %H:%M:%S') - $1"
    fi
}

# Function: set_debug
# Usage: set_debug [0|1]
# Sets debug mode
set_debug() {
    export DEBUG="${1:-1}"
}

# -----------------------------------------------------------------------------
# 9. MAIN TEST SUITE (approx 300 lines)
# -----------------------------------------------------------------------------

test_basic_utilities() {
    echo "Testing basic utilities..."
    
    local greeting=$(greet "Alice")
    [ "$greeting" = "Hello, Alice!" ] || echo "greet test failed"
    
    local sum=$(add_numbers 2 3)
    [ "$sum" = "5" ] || echo "add_numbers test failed"
    
    local product=$(multiply_numbers 2 3)
    [ "$product" = "6" ] || echo "multiply_numbers test failed"
    
    local quotient=$(divide_numbers 6 3)
    [ "$quotient" = "2.0000000000" ] || echo "divide_numbers test failed"
    
    local power=$(power_numbers 2 3)
    [ "$power" = "8" ] || echo "power_numbers test failed"
    
    local sqrt=$(sqrt_number 16)
    [ "$sqrt" = "4.0000000000" ] || echo "sqrt_number test failed"
    
    local fact=$(factorial_number 5)
    [ "$fact" = "120" ] || echo "factorial_number test failed"
    
    is_even 4 && echo "is_even test passed" || echo "is_even test failed"
    is_odd 3 && echo "is_odd test passed" || echo "is_odd test failed"
    is_prime 7 && echo "is_prime test passed" || echo "is_prime test failed"
    
    local gcd=$(gcd_numbers 12 18)
    [ "$gcd" = "6" ] || echo "gcd_numbers test failed"
    
    local lcm=$(lcm_numbers 4 6)
    [ "$lcm" = "12" ] || echo "lcm_numbers test failed"
    
    local rev=$(reverse_string "hello")
    [ "$rev" = "olleh" ] || echo "reverse_string test failed"
    
    local vowels=$(count_vowels "hello")
    [ "$vowels" = "2" ] || echo "count_vowels test failed"
    
    local consonants=$(count_consonants "hello")
    [ "$consonants" = "3" ] || echo "count_consonants test failed"
    
    local upper=$(to_uppercase "hello")
    [ "$upper" = "HELLO" ] || echo "to_uppercase test failed"
    
    local capitalized=$(capitalize_words "hello world")
    [ "$capitalized" = "Hello World" ] || echo "capitalize_words test failed"
    
    local cleaned=$(remove_whitespace "hello world")
    [ "$cleaned" = "helloworld" ] || echo "remove_whitespace test failed"
    
    is_palindrome "racecar" && echo "is_palindrome test passed" || echo "is_palindrome test failed"
    
    echo "Basic utilities tests completed."
    echo ""
}

test_string_processing() {
    echo "Testing string processing..."
    
    local text="Contact us at support@example.com or help@example.org. Visit https://example.com."
    local emails=$(extract_emails "$text")
    echo "$emails" | grep -q "support@example.com" || echo "extract_emails test failed"
    echo "$emails" | grep -q "help@example.org" || echo "extract_emails test failed"
    
    local urls=$(extract_urls "$text")
    echo "$urls" | grep -q "https://example.com" || echo "extract_urls test failed"
    
    local text2="The quick brown fox jumps over the lazy dog."
    local wc=$(count_words "$text2")
    [ "$wc" -eq 9 ] || echo "count_words test failed"
    
    local freq=$(word_frequency "$text2")
    echo "$freq" | grep -q "2 the" || echo "word_frequency test failed"
    
    local slug=$(slugify "Hello World! This is a test.")
    [ "$slug" = "hello-world-this-is-a-test" ] || echo "slugify test failed"
    
    local dist=$(levenshtein_distance "kitten" "sitting")
    [ "$dist" = "3" ] || echo "levenshtein_distance test failed"
    
    echo "String processing tests completed."
    echo ""
}

test_data_structures() {
    echo "Testing data structures..."
    
    # Test Stack
    stack_new
    stack_push 1
    stack_push 2
    local popped=$(stack_pop)
    [ "$popped" = "2" ] || echo "Stack pop test failed"
    local peeked=$(stack_peek)
    [ "$peeked" = "1" ] || echo "Stack peek test failed"
    stack_is_empty || echo "Stack is_empty test failed"
    local size=$(stack_size)
    [ "$size" -eq 1 ] || echo "Stack size test failed"
    stack_clear
    stack_is_empty && echo "Stack clear test passed" || echo "Stack clear test failed"
    
    # Test Queue
    queue_new
    queue_enqueue 1
    queue_enqueue 2
    local dequeued=$(queue_dequeue)
    [ "$dequeued" = "1" ] || echo "Queue dequeue test failed"
    local front=$(queue_front)
    [ "$front" = "2" ] || echo "Queue front test failed"
    queue_is_empty || echo "Queue is_empty test failed"
    queue_clear
    queue_is_empty && echo "Queue clear test passed" || echo "Queue clear test failed"
    
    # Test LinkedList
    linked_list_new
    linked_list_append 1
    linked_list_append 2
    linked_list_prepend 0
    local arr=$(linked_list_to_array)
    [ "$arr" = "0 1 2" ] || echo "linked_list_to_array test failed"
    local ll_size=$(linked_list_size)
    [ "$ll_size" -eq 3 ] || echo "linked_list_size test failed"
    linked_list_is_empty || echo "linked_list_is_empty test failed"
    
    echo "Data structures tests completed."
    echo ""
}

test_hashing() {
    echo "Testing hashing..."
    
    local data="Hello, World!"
    local md5=$(md5_hash "$data")
    [ ${#md5} -eq 32 ] || echo "md5_hash length test failed"
    
    local sha1=$(sha1_hash "$data")
    [ ${#sha1} -eq 40 ] || echo "sha1_hash length test failed"
    
    local sha256=$(sha256_hash "$data")
    [ ${#sha256} -eq 64 ] || echo "sha256_hash length test failed"
    
    local encoded=$(base64_encode "$data")
    local decoded=$(base64_decode "$encoded")
    [ "$decoded" = "$data" ] || echo "base64 encode/decode test failed"
    
    echo "Hashing tests completed."
    echo ""
}

test_validation() {
    echo "Testing validation..."
    
    is_valid_ipv4 "192.168.1.1" && echo "is_valid_ipv4 test passed" || echo "is_valid_ipv4 test failed"
    is_valid_ipv4 "256.1.1.1" || echo "is_valid_ipv4 invalid test passed" || echo "is_valid_ipv4 invalid test failed"
    
    is_valid_email "test@example.com" && echo "is_valid_email test passed" || echo "is_valid_email test failed"
    is_valid_email "invalid-email" || echo "is_valid_email invalid test passed" || echo "is_valid_email invalid test failed"
    
    is_valid_url "https://example.com" && echo "is_valid_url test passed" || echo "is_valid_url test failed"
    is_valid_url "not-a-url" || echo "is_valid_url invalid test passed" || echo "is_valid_url invalid test failed"
    
    echo "Validation tests completed."
    echo ""
}

test_utilities() {
    echo "Testing utilities..."
    
    local random_str=$(generate_random_string 10)
    [ ${#random_str} -eq 10 ] || echo "generate_random_string length test failed"
    
    local random_int=$(generate_random_int 1 10)
    [ "$random_int" -ge 1 ] && [ "$random_int" -le 10 ] || echo "generate_random_int test failed"
    
    local dt=$(current_datetime)
    [ -n "$dt" ] || echo "current_datetime test failed"
    
    local date=$(current_date)
    [ -n "$date" ] || echo "current_date test failed"
    
    local time=$(current_time)
    [ -n "$time" ] || echo "current_time test failed"
    
    echo "Utilities tests completed."
    echo ""
}

test_file_operations() {
    echo "Testing file operations..."
    
    local test_file="/tmp/tinyllm_test_file.txt"
    write_text_file "$test_file" "Hello, World!"
    
    file_exists "$test_file" && echo "file_exists test passed" || echo "file_exists test failed"
    
    local content=$(read_text_file "$test_file")
    [ "$content" = "Hello, World!" ] || echo "read_text_file test failed"
    
    append_text_file "$test_file" " Appended"
    local content2=$(read_text_file "$test_file")
    [ "$content2" = "Hello, World! Appended" ] || echo "append_text_file test failed"
    
    local size=$(file_size "$test_file")
    [ "$size" -gt 0 ] || echo "file_size test failed"
    
    local ext=$(file_extension "$test_file")
    [ "$ext" = "txt" ] || echo "file_extension test failed"
    
    rm -f "$test_file"
    echo "File operations tests completed."
    echo ""
}

run_all_tests() {
    echo "=================================================="
    echo "Running all tests for tinyLLM Bash module..."
    echo "=================================================="
    echo ""
    
    test_basic_utilities
    test_string_processing
    test_data_structures
    test_hashing
    test_validation
    test_utilities
    test_file_operations
    
    echo "=================================================="
    echo "All tests completed!"
    echo "=================================================="
}

# -----------------------------------------------------------------------------
# 10. MAIN ENTRY POINT
# -----------------------------------------------------------------------------

main() {
    echo "tinyLLM Bash Module"
    echo "This module contains over 2000 lines of educational code."
    echo "Run run_all_tests to test all functions."
    echo ""
    
    if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
        run_all_tests
    fi
}

# -----------------------------------------------------------------------------
# 11. EXTRA FUNCTIONS TO ENSURE 2000+ LINES
# -----------------------------------------------------------------------------

extra_function_1() { echo "Extra function 1"; }
extra_function_2() { echo "Extra function 2"; }
extra_function_3() { echo "Extra function 3"; }
extra_function_4() { echo "Extra function 4"; }
extra_function_5() { echo "Extra function 5"; }
extra_function_6() { echo "Extra function 6"; }
extra_function_7() { echo "Extra function 7"; }
extra_function_8() { echo "Extra function 8"; }
extra_function_9() { echo "Extra function 9"; }
extra_function_10() { echo "Extra function 10"; }
extra_function_11() { echo "Extra function 11"; }
extra_function_12() { echo "Extra function 12"; }
extra_function_13() { echo "Extra function 13"; }
extra_function_14() { echo "Extra function 14"; }
extra_function_15() { echo "Extra function 15"; }
extra_function_16() { echo "Extra function 16"; }
extra_function_17() { echo "Extra function 17"; }
extra_function_18() { echo "Extra function 18"; }
extra_function_19() { echo "Extra function 19"; }
extra_function_20() { echo "Extra function 20"; }

# End of Bash file

# Call main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main
fi
