import matplotlib.font_manager as fm

# 找出所有支持中文的字体名称
chinese_fonts = []
for font in fm.fontManager.ttflist:
    if 'CJK' in font.name or 'Hei' in font.name or 'Kai' in font.name or 'Song' in font.name:
        chinese_fonts.append(font.name)
print(set(chinese_fonts))