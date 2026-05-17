
git能够读到的，体积前10的文件：
git ls-files | xargs du -h | sort -hr | head -10