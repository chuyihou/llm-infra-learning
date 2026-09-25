# LLM Infra 深潜 · Deep Dive

学 LLM Infra 的公开笔记。两种内容，一个 repo：

## notes/ — 学习笔记（主力）

学习过程中的原始记录，学到哪写到哪。唯一的约定是每篇开头三行：日期 / 主题 / 状态（学习中 · 已验证 · 存疑）。没有模板，没有流程。

## deep-dives/ — 旗舰长文（偶尔）

某个主题学透了，值得对着源码讲一次，就做成 deep dive：一个交互页面 + 代码级验证。
硬标准只有三条：关键 claim 标 repo + commit + 行号；验证基线写死；未验证的单列。
新开一期见 `deep-dives/template/`。

## 目录

| 期号 | 主题 | 状态 |
|------|------|------|
| deep-dives/01 | Chunked Prefill：把长 Prompt 变成可调度的工作 | ✅ 已发布 |
| notes/2026-09-19 | LLM RL 框架：slime、verl、Relax、Miles、OpenRLHF | 学习中 |

选题顺序见 [ROADMAP.md](ROADMAP.md)（只排 deep dive，notes 不排期）。

## 发布

写完 → 推 GitHub。想发哪个社区，就从同一份 markdown 简单改写发过去，不用搞复杂流程。
各平台只是把同一份笔记换个包装：知乎/HF 放全文，X 拆成 thread，小红书做成卡片。
