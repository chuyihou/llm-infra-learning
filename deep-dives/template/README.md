# 新开一期 deep dive：操作手册

> 注意：这个模板只给 `deep-dives/` 用。`notes/` 不用模板，学到哪写到哪。

## 步骤

1. 复制模板建目录：
   ```
   cp -r deep-dives/template/ deep-dives/NN-topic-slug/
   ```
   编号按 ROADMAP.md 顺延，slug 用英文短横线命名（例：`02-continuous-batching`）。

2. 填 `NOTES.md`：markdown 源文件，先写完它再做交互页。

3. 填 `VERIFICATION.md`：验证基线（哪天、对着哪个 commit 验证的）+ 未验证/存疑清单。

4. 做 `index.html`：复制上一期的交互页作为起点，保留视觉语言，只换内容。
   不要从零写页面——视觉一致性就是品牌。

5. 更新根目录 `README.md` 的目录表和 `ROADMAP.md` 的状态，然后发布。

## 质量门禁（缺一条就不发）

- [ ] 关键实现细节都有 commit + 文件路径 + 行号
- [ ] VERIFICATION.md 里验证基线已填，未验证项已单列
- [ ] 自己能回答"这期里哪个结论如果代码更新了会失效"
