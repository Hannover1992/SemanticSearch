
for file in ./txt/*.txt; do
    pandoc "$file" -o "${file%.txt}.pdf"
done

mv ./txt/*.pdf ./papers/
