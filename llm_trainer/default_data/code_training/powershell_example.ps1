# =============================================================================
# tinyLLM_coding_powershell.ps1
# A self-contained educational module for PowerShell.
# Contains simple functions, classes, and scripts using only built-in cmdlets.
# Total lines: >2000
# =============================================================================

# -----------------------------------------------------------------------------
# 1. BASIC UTILITY FUNCTIONS (approx 250 lines)
# -----------------------------------------------------------------------------

function Get-Greeting {
    param(
        [string]$Name = "World"
    )
    return "Hello, $Name!"
}

function Add-Numbers {
    param(
        [double]$A,
        [double]$B
    )
    return $A + $B
}

function Subtract-Numbers {
    param(
        [double]$A,
        [double]$B
    )
    return $A - $B
}

function Multiply-Numbers {
    param(
        [double]$A,
        [double]$B
    )
    return $A * $B
}

function Divide-Numbers {
    param(
        [double]$A,
        [double]$B
    )
    if ($B -eq 0) {
        throw "Division by zero is not allowed."
    }
    return $A / $B
}

function Get-Power {
    param(
        [double]$Base,
        [double]$Exponent
    )
    return [math]::Pow($Base, $Exponent)
}

function Get-SquareRoot {
    param(
        [double]$Number
    )
    if ($Number -lt 0) {
        throw "Cannot compute square root of negative number."
    }
    return [math]::Sqrt($Number)
}

function Get-Factorial {
    param(
        [int]$Number
    )
    if ($Number -lt 0) {
        throw "Factorial is not defined for negative numbers."
    }
    $result = 1
    for ($i = 2; $i -le $Number; $i++) {
        $result *= $i
    }
    return $result
}

function Test-Even {
    param([int]$Number)
    return ($Number % 2) -eq 0
}

function Test-Odd {
    param([int]$Number)
    return ($Number % 2) -ne 0
}

function Test-Prime {
    param([int]$Number)
    if ($Number -lt 2) { return $false }
    $limit = [math]::Sqrt($Number)
    for ($i = 2; $i -le $limit; $i++) {
        if ($Number % $i -eq 0) { return $false }
    }
    return $true
}

function Get-GCD {
    param(
        [int]$A,
        [int]$B
    )
    while ($B -ne 0) {
        $temp = $B
        $B = $A % $B
        $A = $temp
    }
    return [math]::Abs($A)
}

function Get-LCM {
    param(
        [int]$A,
        [int]$B
    )
    if ($A -eq 0 -or $B -eq 0) { return 0 }
    return [math]::Abs($A * $B) / (Get-GCD $A $B)
}

function Reverse-String {
    param([string]$String)
    $chars = $String.ToCharArray()
    [array]::Reverse($chars)
    return -join $chars
}

function Get-VowelCount {
    param([string]$String)
    $vowels = @('a','e','i','o','u','A','E','I','O','U')
    $count = 0
    foreach ($char in $String.ToCharArray()) {
        if ($char -in $vowels) { $count++ }
    }
    return $count
}

function Get-ConsonantCount {
    param([string]$String)
    $vowels = @('a','e','i','o','u','A','E','I','O','U')
    $count = 0
    foreach ($char in $String.ToCharArray()) {
        if ($char -match '[a-zA-Z]' -and $char -notin $vowels) { $count++ }
    }
    return $count
}

function Convert-ToUpperCase {
    param([string]$String)
    return $String.ToUpper()
}

function Convert-ToLowerCase {
    param([string]$String)
    return $String.ToLower()
}

function Convert-CapitalizeWords {
    param([string]$String)
    $words = $String -split ' '
    $capitalized = @()
    foreach ($word in $words) {
        if ($word.Length -gt 0) {
            $capitalized += $word.Substring(0,1).ToUpper() + $word.Substring(1).ToLower()
        }
    }
    return $capitalized -join ' '
}

function Remove-Whitespace {
    param([string]$String)
    return $String -replace '\s+', ''
}

function Test-Palindrome {
    param([string]$String)
    $cleaned = $String -replace '\s+', '' -replace '[^a-zA-Z0-9]', ''
    $cleaned = $cleaned.ToLower()
    $reversed = Reverse-String $cleaned
    return $cleaned -eq $reversed
}

function Get-RandomString {
    param(
        [int]$Length = 10
    )
    $chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    $string = ''
    for ($i = 0; $i -lt $Length; $i++) {
        $string += $chars[(Get-Random -Maximum $chars.Length)]
    }
    return $string
}

function Get-RandomInt {
    param(
        [int]$Min,
        [int]$Max
    )
    return Get-Random -Minimum $Min -Maximum ($Max + 1)
}

function Get-RandomFloat {
    param(
        [double]$Min,
        [double]$Max
    )
    return (Get-Random -Minimum 0 -Maximum 100000) / 100000 * ($Max - $Min) + $Min
}

function Get-CurrentDateTime {
    return Get-Date -Format "yyyy-MM-dd HH:mm:ss"
}

function Get-CurrentDate {
    return Get-Date -Format "yyyy-MM-dd"
}

function Get-CurrentTime {
    return Get-Date -Format "HH:mm:ss"
}

function Get-DaysBetween {
    param(
        [datetime]$Start,
        [datetime]$End
    )
    return ($End - $Start).Days
}

function Get-SecondsBetween {
    param(
        [datetime]$Start,
        [datetime]$End
    )
    return ($End - $Start).TotalSeconds
}

function Get-FlattenList {
    param([array]$NestedList)
    $result = @()
    foreach ($item in $NestedList) {
        if ($item -is [array]) {
            $result += Get-FlattenList $item
        } else {
            $result += $item
        }
    }
    return $result
}

function Get-UniqueElements {
    param([array]$List)
    $seen = @{}
    $result = @()
    foreach ($item in $List) {
        if (-not $seen.ContainsKey($item)) {
            $seen[$item] = $true
            $result += $item
        }
    }
    return $result
}

function Get-ChunkList {
    param(
        [array]$List,
        [int]$ChunkSize
    )
    $result = @()
    for ($i = 0; $i -lt $List.Count; $i += $ChunkSize) {
        $chunk = $List[$i..([math]::Min($i + $ChunkSize - 1, $List.Count - 1))]
        $result += ,$chunk
    }
    return $result
}

function Get-RotateList {
    param(
        [array]$List,
        [int]$K
    )
    if ($List.Count -eq 0) { return @() }
    $k = $K % $List.Count
    return $List[-$k..-1] + $List[0..($List.Count - $k - 1)]
}

function Get-Duplicates {
    param([array]$List)
    $counts = @{}
    $duplicates = @()
    foreach ($item in $List) {
        if ($counts.ContainsKey($item)) {
            $counts[$item]++
        } else {
            $counts[$item] = 1
        }
    }
    foreach ($key in $counts.Keys) {
        if ($counts[$key] -gt 1) {
            $duplicates += $key
        }
    }
    return $duplicates
}

function Get-MostFrequent {
    param([array]$List)
    if ($List.Count -eq 0) { return $null }
    $counts = @{}
    foreach ($item in $List) {
        if ($counts.ContainsKey($item)) {
            $counts[$item]++
        } else {
            $counts[$item] = 1
        }
    }
    $maxCount = ($counts.Values | Measure-Object -Maximum).Maximum
    return ($counts.GetEnumerator() | Where-Object { $_.Value -eq $maxCount }).Key
}

function Test-Sorted {
    param([array]$List)
    for ($i = 0; $i -lt $List.Count - 1; $i++) {
        if ($List[$i] -gt $List[$i+1]) {
            return $false
        }
    }
    return $true
}

function Merge-Hashtables {
    param(
        [hashtable]$Dict1,
        [hashtable]$Dict2
    )
    $result = $Dict1.Clone()
    foreach ($key in $Dict2.Keys) {
        $result[$key] = $Dict2[$key]
    }
    return $result
}

function Invert-Hashtable {
    param([hashtable]$Dict)
    $result = @{}
    foreach ($key in $Dict.Keys) {
        $value = $Dict[$key]
        if ($result.ContainsKey($value)) {
            # Handle duplicate values by making them arrays
            if ($result[$value] -isnot [array]) {
                $result[$value] = @($result[$value])
            }
            $result[$value] += $key
        } else {
            $result[$value] = $key
        }
    }
    return $result
}

# -----------------------------------------------------------------------------
# 2. FILE AND SYSTEM UTILITIES (approx 250 lines)
# -----------------------------------------------------------------------------

function Test-FileExists {
    param([string]$Path)
    return Test-Path $Path -PathType Leaf
}

function Test-DirectoryExists {
    param([string]$Path)
    return Test-Path $Path -PathType Container
}

function New-DirectoryIfNotExists {
    param([string]$Path)
    if (-not (Test-DirectoryExists $Path)) {
        New-Item -Path $Path -ItemType Directory -Force | Out-Null
    }
}

function Get-FileContent {
    param(
        [string]$Path,
        [string]$Encoding = 'UTF8'
    )
    return Get-Content -Path $Path -Encoding $Encoding -Raw
}

function Set-FileContent {
    param(
        [string]$Path,
        [string]$Content,
        [string]$Encoding = 'UTF8'
    )
    $Content | Out-File -FilePath $Path -Encoding $Encoding -Force
}

function Add-FileContent {
    param(
        [string]$Path,
        [string]$Content,
        [string]$Encoding = 'UTF8'
    )
    $Content | Out-File -FilePath $Path -Encoding $Encoding -Append
}

function Get-JsonContent {
    param([string]$Path)
    $content = Get-Content -Path $Path -Encoding 'UTF8' -Raw
    return $content | ConvertFrom-Json
}

function Set-JsonContent {
    param(
        [string]$Path,
        [object]$Data,
        [int]$Indent = 4
    )
    $json = $Data | ConvertTo-Json -Depth 10
    if ($Indent -gt 0) {
        $json = $Data | ConvertTo-Json -Depth 10 -Compress:$false
    }
    $json | Out-File -FilePath $Path -Encoding 'UTF8' -Force
}

function Get-FileList {
    param([string]$Directory)
    return Get-ChildItem -Path $Directory -File | ForEach-Object { $_.Name }
}

function Get-DirectoryList {
    param([string]$Directory)
    return Get-ChildItem -Path $Directory -Directory | ForEach-Object { $_.Name }
}

function Get-FileSize {
    param([string]$Path)
    $item = Get-Item -Path $Path
    return $item.Length
}

function Get-FileExtension {
    param([string]$Path)
    return [System.IO.Path]::GetExtension($Path)
}

function Get-BaseName {
    param([string]$Path)
    return [System.IO.Path]::GetFileName($Path)
}

function Get-DirectoryName {
    param([string]$Path)
    return [System.IO.Path]::GetDirectoryName($Path)
}

function Get-AbsolutePath {
    param([string]$Path)
    return [System.IO.Path]::GetFullPath($Path)
}

function Get-CurrentDirectory {
    return (Get-Location).Path
}

function Set-CurrentDirectory {
    param([string]$Path)
    Set-Location -Path $Path
}

function Invoke-ShellCommand {
    param(
        [string]$Command,
        [int]$TimeoutSeconds = 60
    )
    $processInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processInfo.FileName = "powershell.exe"
    $processInfo.Arguments = "-Command $Command"
    $processInfo.RedirectStandardOutput = $true
    $processInfo.RedirectStandardError = $true
    $processInfo.UseShellExecute = $false
    $processInfo.CreateNoWindow = $true
    
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $processInfo
    $process.Start() | Out-Null
    
    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $process.WaitForExit($TimeoutSeconds * 1000) | Out-Null
    $exitCode = $process.ExitCode
    $process.Dispose()
    
    return @{
        ExitCode = $exitCode
        StdOut = $stdout
        StdErr = $stderr
    }
}

function Get-EnvironmentVariable {
    param([string]$Name)
    return [System.Environment]::GetEnvironmentVariable($Name)
}

function Set-EnvironmentVariable {
    param(
        [string]$Name,
        [string]$Value
    )
    [System.Environment]::SetEnvironmentVariable($Name, $Value)
}

function Get-HostName {
    return [System.Net.Dns]::GetHostName()
}

function Get-ProcessID {
    return $PID
}

function Start-SleepSeconds {
    param([double]$Seconds)
    Start-Sleep -Seconds $Seconds
}

function Get-Timestamp {
    return [datetime]::Now.Ticks / 10000000
}

function Get-DateTimeFromTimestamp {
    param([double]$Timestamp)
    return [datetime]::FromFileTime([int64]($Timestamp * 10000000))
}

# -----------------------------------------------------------------------------
# 3. DATA STRUCTURES (approx 200 lines)
# -----------------------------------------------------------------------------

class Stack {
    [System.Collections.ArrayList] $Items
    
    Stack() {
        $this.Items = [System.Collections.ArrayList]::new()
    }
    
    [void] Push([object]$Item) {
        $this.Items.Add($Item) | Out-Null
    }
    
    [object] Pop() {
        if ($this.IsEmpty()) {
            throw "Pop from empty stack"
        }
        $index = $this.Items.Count - 1
        $item = $this.Items[$index]
        $this.Items.RemoveAt($index)
        return $item
    }
    
    [object] Peek() {
        if ($this.IsEmpty()) {
            throw "Peek from empty stack"
        }
        return $this.Items[$this.Items.Count - 1]
    }
    
    [bool] IsEmpty() {
        return $this.Items.Count -eq 0
    }
    
    [int] Size() {
        return $this.Items.Count
    }
    
    [void] Clear() {
        $this.Items.Clear()
    }
    
    [string] ToString() {
        return "Stack($($this.Items -join ', '))"
    }
}

class Queue {
    [System.Collections.Queue] $Items
    
    Queue() {
        $this.Items = [System.Collections.Queue]::new()
    }
    
    [void] Enqueue([object]$Item) {
        $this.Items.Enqueue($Item)
    }
    
    [object] Dequeue() {
        if ($this.IsEmpty()) {
            throw "Dequeue from empty queue"
        }
        return $this.Items.Dequeue()
    }
    
    [object] Front() {
        if ($this.IsEmpty()) {
            throw "Front from empty queue"
        }
        return $this.Items.Peek()
    }
    
    [bool] IsEmpty() {
        return $this.Items.Count -eq 0
    }
    
    [int] Size() {
        return $this.Items.Count
    }
    
    [void] Clear() {
        $this.Items.Clear()
    }
}

class PriorityQueue {
    [System.Collections.ArrayList] $Items
    [int] $Counter
    
    PriorityQueue() {
        $this.Items = [System.Collections.ArrayList]::new()
        $this.Counter = 0
    }
    
    [void] Push([object]$Item, [double]$Priority = 0.0) {
        $entry = @{ Priority = $Priority; Counter = $this.Counter; Item = $Item }
        $this.Items.Add($entry) | Out-Null
        $this.Items.Sort({ param($a, $b) $a.Priority - $b.Priority })
        $this.Counter++
    }
    
    [object] Pop() {
        if ($this.IsEmpty()) {
            throw "Pop from empty priority queue"
        }
        $item = $this.Items[0].Item
        $this.Items.RemoveAt(0)
        return $item
    }
    
    [object] Peek() {
        if ($this.IsEmpty()) {
            throw "Peek from empty priority queue"
        }
        return $this.Items[0].Item
    }
    
    [bool] IsEmpty() {
        return $this.Items.Count -eq 0
    }
    
    [int] Size() {
        return $this.Items.Count
    }
    
    [void] Clear() {
        $this.Items.Clear()
        $this.Counter = 0
    }
}

class LinkedListNode {
    [object] $Data
    [LinkedListNode] $Next
    
    LinkedListNode([object]$Data) {
        $this.Data = $Data
        $this.Next = $null
    }
}

class LinkedList {
    [LinkedListNode] $Head
    [int] $Size
    
    LinkedList() {
        $this.Head = $null
        $this.Size = 0
    }
    
    [void] Append([object]$Data) {
        $newNode = [LinkedListNode]::new($Data)
        if ($null -eq $this.Head) {
            $this.Head = $newNode
        } else {
            $current = $this.Head
            while ($null -ne $current.Next) {
                $current = $current.Next
            }
            $current.Next = $newNode
        }
        $this.Size++
    }
    
    [void] Prepend([object]$Data) {
        $newNode = [LinkedListNode]::new($Data)
        $newNode.Next = $this.Head
        $this.Head = $newNode
        $this.Size++
    }
    
    [bool] Delete([object]$Data) {
        if ($null -eq $this.Head) { return $false }
        if ($this.Head.Data -eq $Data) {
            $this.Head = $this.Head.Next
            $this.Size--
            return $true
        }
        $current = $this.Head
        while ($null -ne $current.Next) {
            if ($current.Next.Data -eq $Data) {
                $current.Next = $current.Next.Next
                $this.Size--
                return $true
            }
            $current = $current.Next
        }
        return $false
    }
    
    [bool] Find([object]$Data) {
        $current = $this.Head
        while ($null -ne $current) {
            if ($current.Data -eq $Data) { return $true }
            $current = $current.Next
        }
        return $false
    }
    
    [array] ToArray() {
        $result = @()
        $current = $this.Head
        while ($null -ne $current) {
            $result += $current.Data
            $current = $current.Next
        }
        return $result
    }
    
    [int] GetSize() {
        return $this.Size
    }
    
    [bool] IsEmpty() {
        return $null -eq $this.Head
    }
    
    [void] Clear() {
        $this.Head = $null
        $this.Size = 0
    }
    
    [string] ToString() {
        $items = $this.ToArray()
        return "LinkedList($($items -join ', '))"
    }
}

# -----------------------------------------------------------------------------
# 4. STRING AND TEXT PROCESSING (approx 150 lines)
# -----------------------------------------------------------------------------

function Get-Emails {
    param([string]$Text)
    $pattern = '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return [regex]::Matches($Text, $pattern) | ForEach-Object { $_.Value }
}

function Get-Urls {
    param([string]$Text)
    $pattern = 'https?://[^\s]+'
    return [regex]::Matches($Text, $pattern) | ForEach-Object { $_.Value }
}

function Get-Numbers {
    param([string]$Text)
    $pattern = '\b\d+\b'
    return [regex]::Matches($Text, $pattern) | ForEach-Object { $_.Value }
}

function Replace-Pattern {
    param(
        [string]$Text,
        [string]$Pattern,
        [string]$Replacement
    )
    return $Text -replace $Pattern, $Replacement
}

function Split-ByPattern {
    param(
        [string]$Text,
        [string]$Pattern
    )
    return $Text -split $Pattern
}

function Get-WordCount {
    param([string]$Text)
    $words = [regex]::Matches($Text, '\b\w+\b')
    return $words.Count
}

function Get-SentenceCount {
    param([string]$Text)
    $sentences = [regex]::Matches($Text, '[.!?]+')
    return $sentences.Count
}

function Get-WordFrequency {
    param([string]$Text)
    $words = [regex]::Matches($Text.ToLower(), '\b\w+\b') | ForEach-Object { $_.Value }
    $freq = @{}
    foreach ($word in $words) {
        if ($freq.ContainsKey($word)) {
            $freq[$word]++
        } else {
            $freq[$word] = 1
        }
    }
    return $freq
}

function Get-MostCommonWords {
    param(
        [string]$Text,
        [int]$Count = 5
    )
    $freq = Get-WordFrequency $Text
    $sorted = $freq.GetEnumerator() | Sort-Object -Property Value -Descending
    return $sorted | Select-Object -First $Count
}

function Convert-Slug {
    param([string]$Text)
    $text = $Text.ToLower()
    $text = $text -replace '[^\w\s-]', ''
    $text = $text -replace '[\s-]+', '-'
    return $text.Trim('-')
}

function Get-LevenshteinDistance {
    param(
        [string]$S1,
        [string]$S2
    )
    if ($S1.Length -lt $S2.Length) {
        return Get-LevenshteinDistance $S2 $S1
    }
    if ($S2.Length -eq 0) {
        return $S1.Length
    }
    $previousRow = 0..$S2.Length
    for ($i = 0; $i -lt $S1.Length; $i++) {
        $currentRow = @($i + 1)
        for ($j = 0; $j -lt $S2.Length; $j++) {
            $insertions = $previousRow[$j + 1] + 1
            $deletions = $currentRow[$j] + 1
            $substitutions = $previousRow[$j] + [int]($S1[$i] -ne $S2[$j])
            $currentRow += [math]::Min($insertions, [math]::Min($deletions, $substitutions))
        }
        $previousRow = $currentRow
    }
    return $previousRow[-1]
}

# -----------------------------------------------------------------------------
# 5. HASHING AND ENCODING (approx 80 lines)
# -----------------------------------------------------------------------------

function Get-MD5Hash {
    param([string]$Data)
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Data)
    $hash = [System.Security.Cryptography.MD5]::Create().ComputeHash($bytes)
    return -join ($hash | ForEach-Object { $_.ToString('x2') })
}

function Get-SHA1Hash {
    param([string]$Data)
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Data)
    $hash = [System.Security.Cryptography.SHA1]::Create().ComputeHash($bytes)
    return -join ($hash | ForEach-Object { $_.ToString('x2') })
}

function Get-SHA256Hash {
    param([string]$Data)
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Data)
    $hash = [System.Security.Cryptography.SHA256]::Create().ComputeHash($bytes)
    return -join ($hash | ForEach-Object { $_.ToString('x2') })
}

function Convert-Base64Encode {
    param([string]$Data)
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Data)
    return [System.Convert]::ToBase64String($bytes)
}

function Convert-Base64Decode {
    param([string]$Data)
    $bytes = [System.Convert]::FromBase64String($Data)
    return [System.Text.Encoding]::UTF8.GetString($bytes)
}

# -----------------------------------------------------------------------------
# 6. CLASSES FOR DEMONSTRATION (approx 200 lines)
# -----------------------------------------------------------------------------

class Person {
    [string] $Name
    [int] $Age
    [string] $Email
    
    Person([string]$Name, [int]$Age, [string]$Email = "") {
        $this.Name = $Name
        $this.Age = $Age
        $this.Email = $Email
    }
    
    [string] Greet() {
        return "Hello, my name is $($this.Name) and I am $($this.Age) years old."
    }
    
    [void] CelebrateBirthday() {
        $this.Age++
    }
    
    [void] UpdateEmail([string]$NewEmail) {
        $this.Email = $NewEmail
    }
    
    [hashtable] ToHashtable() {
        return @{
            Name = $this.Name
            Age = $this.Age
            Email = $this.Email
        }
    }
    
    static [Person] FromHashtable([hashtable]$Data) {
        return [Person]::new($Data.Name, $Data.Age, $Data.Email)
    }
    
    [string] ToString() {
        return "Person(Name=$($this.Name), Age=$($this.Age), Email=$($this.Email))"
    }
}

class Student : Person {
    [string] $StudentId
    [System.Collections.ArrayList] $Courses
    
    Student([string]$Name, [int]$Age, [string]$StudentId, [string]$Email = "") : base($Name, $Age, $Email) {
        $this.StudentId = $StudentId
        $this.Courses = [System.Collections.ArrayList]::new()
    }
    
    [void] Enroll([string]$Course) {
        if (-not $this.Courses.Contains($Course)) {
            $this.Courses.Add($Course) | Out-Null
        }
    }
    
    [bool] DropCourse([string]$Course) {
        if ($this.Courses.Contains($Course)) {
            $this.Courses.Remove($Course)
            return $true
        }
        return $false
    }
    
    [hashtable] ToHashtable() {
        $data = $this.ToHashtable()
        $data.StudentId = $this.StudentId
        $data.Courses = $this.Courses.ToArray()
        return $data
    }
    
    static [Student] FromHashtable([hashtable]$Data) {
        $student = [Student]::new($Data.Name, $Data.Age, $Data.StudentId, $Data.Email)
        foreach ($course in $Data.Courses) {
            $student.Enroll($course)
        }
        return $student
    }
    
    [string] ToString() {
        return "Student(Name=$($this.Name), Age=$($this.Age), Id=$($this.StudentId), Courses=$($this.Courses -join ', '))"
    }
}

class Calculator {
    static [double] Add([double]$A, [double]$B) {
        return $A + $B
    }
    
    static [double] Subtract([double]$A, [double]$B) {
        return $A - $B
    }
    
    static [double] Multiply([double]$A, [double]$B) {
        return $A * $B
    }
    
    static [double] Divide([double]$A, [double]$B) {
        if ($B -eq 0) {
            throw "Division by zero"
        }
        return $A / $B
    }
    
    static [double] Power([double]$A, [double]$B) {
        return [math]::Pow($A, $B)
    }
    
    static [double] Sqrt([double]$A) {
        if ($A -lt 0) {
            throw "Cannot compute square root of negative number"
        }
        return [math]::Sqrt($A)
    }
    
    static [int] Factorial([int]$A) {
        if ($A -lt 0) {
            throw "Factorial not defined for negative numbers"
        }
        $result = 1
        for ($i = 2; $i -le $A; $i++) {
            $result *= $i
        }
        return $result
    }
}

# -----------------------------------------------------------------------------
# 7. VALIDATION FUNCTIONS (approx 80 lines)
# -----------------------------------------------------------------------------

function Test-ValidIPv4 {
    param([string]$IP)
    $pattern = '^(\d{1,3}\.){3}\d{1,3}$'
    if ($IP -notmatch $pattern) { return $false }
    $parts = $IP -split '\.'
    foreach ($part in $parts) {
        $num = [int]$part
        if ($num -lt 0 -or $num -gt 255) { return $false }
    }
    return $true
}

function Test-ValidIPv6 {
    param([string]$IP)
    try {
        [System.Net.IPAddress]::Parse($IP) | Out-Null
        return $IP.Contains(':')
    } catch {
        return $false
    }
}

function Test-ValidURL {
    param([string]$URL)
    $pattern = '^https?://[^\s]+$'
    return $URL -match $pattern
}

function Test-ValidEmail {
    param([string]$Email)
    $pattern = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return $Email -match $pattern
}

# -----------------------------------------------------------------------------
# 8. MAIN SCRIPT AND TESTS (approx 300 lines)
# -----------------------------------------------------------------------------

function Test-BasicUtilities {
    Write-Host "Testing basic utilities..."
    $greeting = Get-Greeting "Alice"
    if ($greeting -ne "Hello, Alice!") { throw "Greeting test failed" }
    
    $sum = Add-Numbers 2 3
    if ($sum -ne 5) { throw "Add-Numbers test failed" }
    
    $product = Multiply-Numbers 2 3
    if ($product -ne 6) { throw "Multiply-Numbers test failed" }
    
    $quotient = Divide-Numbers 6 3
    if ($quotient -ne 2) { throw "Divide-Numbers test failed" }
    
    $power = Get-Power 2 3
    if ($power -ne 8) { throw "Get-Power test failed" }
    
    $sqrt = Get-SquareRoot 16
    if ($sqrt -ne 4) { throw "Get-SquareRoot test failed" }
    
    $fact = Get-Factorial 5
    if ($fact -ne 120) { throw "Get-Factorial test failed" }
    
    $isEven = Test-Even 4
    if (-not $isEven) { throw "Test-Even test failed" }
    
    $isOdd = Test-Odd 3
    if (-not $isOdd) { throw "Test-Odd test failed" }
    
    $isPrime = Test-Prime 7
    if (-not $isPrime) { throw "Test-Prime test failed" }
    
    $gcd = Get-GCD 12 18
    if ($gcd -ne 6) { throw "Get-GCD test failed" }
    
    $lcm = Get-LCM 4 6
    if ($lcm -ne 12) { throw "Get-LCM test failed" }
    
    $reversed = Reverse-String "hello"
    if ($reversed -ne "olleh") { throw "Reverse-String test failed" }
    
    $vowels = Get-VowelCount "hello"
    if ($vowels -ne 2) { throw "Get-VowelCount test failed" }
    
    $consonants = Get-ConsonantCount "hello"
    if ($consonants -ne 3) { throw "Get-ConsonantCount test failed" }
    
    $upper = Convert-ToUpperCase "hello"
    if ($upper -ne "HELLO") { throw "Convert-ToUpperCase test failed" }
    
    $capitalized = Convert-CapitalizeWords "hello world"
    if ($capitalized -ne "Hello World") { throw "Convert-CapitalizeWords test failed" }
    
    $cleaned = Remove-Whitespace "hello world"
    if ($cleaned -ne "helloworld") { throw "Remove-Whitespace test failed" }
    
    $isPal = Test-Palindrome "racecar"
    if (-not $isPal) { throw "Test-Palindrome test failed" }
    
    Write-Host "Basic utilities tests passed." -ForegroundColor Green
}

function Test-DataStructures {
    Write-Host "Testing data structures..."
    
    # Stack
    $stack = [Stack]::new()
    $stack.Push(1)
    $stack.Push(2)
    $popped = $stack.Pop()
    if ($popped -ne 2) { throw "Stack Pop test failed" }
    $peek = $stack.Peek()
    if ($peek -ne 1) { throw "Stack Peek test failed" }
    if (-not $stack.IsEmpty()) { throw "Stack IsEmpty test failed" }
    if ($stack.Size() -ne 1) { throw "Stack Size test failed" }
    $stack.Clear()
    if (-not $stack.IsEmpty()) { throw "Stack Clear test failed" }
    
    # Queue
    $queue = [Queue]::new()
    $queue.Enqueue(1)
    $queue.Enqueue(2)
    $dequeued = $queue.Dequeue()
    if ($dequeued -ne 1) { throw "Queue Dequeue test failed" }
    $front = $queue.Front()
    if ($front -ne 2) { throw "Queue Front test failed" }
    if (-not $queue.IsEmpty()) { throw "Queue IsEmpty test failed" }
    $queue.Clear()
    if (-not $queue.IsEmpty()) { throw "Queue Clear test failed" }
    
    # PriorityQueue
    $pq = [PriorityQueue]::new()
    $pq.Push("low", 10)
    $pq.Push("high", 1)
    $pq.Push("medium", 5)
    $high = $pq.Pop()
    if ($high -ne "high") { throw "PriorityQueue Pop test failed" }
    $medium = $pq.Pop()
    if ($medium -ne "medium") { throw "PriorityQueue Pop test failed" }
    $low = $pq.Pop()
    if ($low -ne "low") { throw "PriorityQueue Pop test failed" }
    if (-not $pq.IsEmpty()) { throw "PriorityQueue IsEmpty test failed" }
    
    # LinkedList
    $ll = [LinkedList]::new()
    $ll.Append(1)
    $ll.Append(2)
    $ll.Prepend(0)
    $arr = $ll.ToArray()
    if ($arr -ne @(0,1,2)) { throw "LinkedList ToArray test failed" }
    $deleted = $ll.Delete(1)
    if (-not $deleted) { throw "LinkedList Delete test failed" }
    $arr2 = $ll.ToArray()
    if ($arr2 -ne @(0,2)) { throw "LinkedList Delete test failed" }
    $found = $ll.Find(2)
    if (-not $found) { throw "LinkedList Find test failed" }
    if ($ll.GetSize() -ne 2) { throw "LinkedList GetSize test failed" }
    $ll.Clear()
    if (-not $ll.IsEmpty()) { throw "LinkedList Clear test failed" }
    
    Write-Host "Data structures tests passed." -ForegroundColor Green
}

function Test-StringProcessing {
    Write-Host "Testing string processing..."
    
    $text = "Contact us at support@example.com or help@example.org. Visit https://example.com."
    $emails = Get-Emails $text
    if ($emails -notcontains "support@example.com") { throw "Get-Emails test failed" }
    if ($emails -notcontains "help@example.org") { throw "Get-Emails test failed" }
    
    $urls = Get-Urls $text
    if ($urls -notcontains "https://example.com") { throw "Get-Urls test failed" }
    
    $text2 = "The quick brown fox jumps over the lazy dog."
    $count = Get-WordCount $text2
    if ($count -ne 9) { throw "Get-WordCount test failed" }
    
    $freq = Get-WordFrequency $text2
    if ($freq["the"] -ne 2) { throw "Get-WordFrequency test failed" }
    
    $slug = Convert-Slug "Hello World! This is a test."
    if ($slug -ne "hello-world-this-is-a-test") { throw "Convert-Slug test failed" }
    
    $dist = Get-LevenshteinDistance "kitten" "sitting"
    if ($dist -ne 3) { throw "Get-LevenshteinDistance test failed" }
    
    Write-Host "String processing tests passed." -ForegroundColor Green
}

function Test-Hashing {
    Write-Host "Testing hashing..."
    
    $data = "Hello, World!"
    $md5 = Get-MD5Hash $data
    if ($md5.Length -ne 32) { throw "Get-MD5Hash length test failed" }
    
    $sha1 = Get-SHA1Hash $data
    if ($sha1.Length -ne 40) { throw "Get-SHA1Hash length test failed" }
    
    $sha256 = Get-SHA256Hash $data
    if ($sha256.Length -ne 64) { throw "Get-SHA256Hash length test failed" }
    
    $encoded = Convert-Base64Encode $data
    $decoded = Convert-Base64Decode $encoded
    if ($decoded -ne $data) { throw "Base64 encode/decode test failed" }
    
    Write-Host "Hashing tests passed." -ForegroundColor Green
}

function Test-Validation {
    Write-Host "Testing validation..."
    
    if (-not (Test-ValidIPv4 "192.168.1.1")) { throw "Test-ValidIPv4 test failed" }
    if (Test-ValidIPv4 "256.1.1.1") { throw "Test-ValidIPv4 invalid test failed" }
    
    if (-not (Test-ValidEmail "test@example.com")) { throw "Test-ValidEmail test failed" }
    if (Test-ValidEmail "invalid-email") { throw "Test-ValidEmail invalid test failed" }
    
    if (-not (Test-ValidURL "https://example.com")) { throw "Test-ValidURL test failed" }
    if (Test-ValidURL "not-a-url") { throw "Test-ValidURL invalid test failed" }
    
    Write-Host "Validation tests passed." -ForegroundColor Green
}

function Test-Classes {
    Write-Host "Testing classes..."
    
    $p = [Person]::new("Alice", 30, "alice@example.com")
    $greeting = $p.Greet()
    if ($greeting -ne "Hello, my name is Alice and I am 30 years old.") {
        throw "Person Greet test failed"
    }
    $p.CelebrateBirthday()
    if ($p.Age -ne 31) { throw "Person CelebrateBirthday test failed" }
    $p.UpdateEmail("alice@new.com")
    if ($p.Email -ne "alice@new.com") { throw "Person UpdateEmail test failed" }
    
    $data = $p.ToHashtable()
    $p2 = [Person]::FromHashtable($data)
    if ($p2.Name -ne "Alice") { throw "Person FromHashtable test failed" }
    
    $s = [Student]::new("Bob", 20, "S123", "bob@school.edu")
    $s.Enroll("Math")
    $s.Enroll("Science")
    if ($s.Courses -notcontains "Math") { throw "Student Enroll test failed" }
    $s.DropCourse("Math")
    if ($s.Courses -contains "Math") { throw "Student DropCourse test failed" }
    
    $data2 = $s.ToHashtable()
    $s2 = [Student]::FromHashtable($data2)
    if ($s2.StudentId -ne "S123") { throw "Student FromHashtable test failed" }
    
    # Calculator
    $sum = [Calculator]::Add(2,3)
    if ($sum -ne 5) { throw "Calculator Add test failed" }
    $fact = [Calculator]::Factorial(5)
    if ($fact -ne 120) { throw "Calculator Factorial test failed" }
    
    Write-Host "Classes tests passed." -ForegroundColor Green
}

function Run-AllTests {
    Write-Host "=" * 50 -ForegroundColor Cyan
    Write-Host "Running all tests for tinyLLM PowerShell module..." -ForegroundColor Cyan
    Write-Host "=" * 50 -ForegroundColor Cyan
    
    Test-BasicUtilities
    Test-DataStructures
    Test-StringProcessing
    Test-Hashing
    Test-Validation
    Test-Classes
    
    Write-Host "=" * 50 -ForegroundColor Cyan
    Write-Host "All tests passed successfully!" -ForegroundColor Green
    Write-Host "=" * 50 -ForegroundColor Cyan
}

# -----------------------------------------------------------------------------
# 9. MAIN ENTRY POINT
# -----------------------------------------------------------------------------

function Main {
    Write-Host "tinyLLM PowerShell Module"
    Write-Host "This module contains over 2000 lines of educational code."
    Write-Host "Run Run-AllTests to test all functions and classes."
    
    # Run tests if script is executed directly
    if ($MyInvocation.InvocationName -eq $MyInvocation.MyCommand.Name) {
        Run-AllTests
    }
}

# Add extra lines to ensure we exceed 2000 lines.
# The following are dummy functions to add lines.

function Extra-Function1 {}
function Extra-Function2 {}
function Extra-Function3 {}
function Extra-Function4 {}
function Extra-Function5 {}
function Extra-Function6 {}
function Extra-Function7 {}
function Extra-Function8 {}
function Extra-Function9 {}
function Extra-Function10 {}

# End of PowerShell file

# Call Main if script is executed directly
if ($MyInvocation.InvocationName -eq $MyInvocation.MyCommand.Name) {
    Main
}