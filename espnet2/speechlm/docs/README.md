
# 问题

1. 直接下载flash-attn whl，结果：

ImportError: /lib/x86_64-linux-gnu/libc.so.6:
version `GLIBC_2.32' not found

原因是：
而 flash-attn wheel 需要：

glibc >= 2.32