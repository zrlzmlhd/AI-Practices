import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'AI-Practices',
  description: 'A Systematic Approach to AI Research & Engineering',

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/logo.svg' }],
    ['meta', { name: 'theme-color', content: '#007AFF' }],
    ['meta', { name: 'author', content: 'zimingttkx' }],
    ['meta', { name: 'keywords', content: 'AI, Machine Learning, Deep Learning, Neural Networks, Computer Vision, NLP, PyTorch, TensorFlow' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'AI-Practices' }],
    ['meta', { property: 'og:description', content: 'A Systematic Approach to AI Research & Engineering' }],
    ['meta', { property: 'og:url', content: 'https://zimingttkx.github.io/AI-Practices/' }],
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    ['meta', { name: 'twitter:title', content: 'AI-Practices' }],
    ['meta', { name: 'twitter:description', content: 'A Systematic Approach to AI Research & Engineering' }],
  ],

  sitemap: {
    hostname: 'https://zimingttkx.github.io/AI-Practices/'
  },

  lastUpdated: true,
  cleanUrls: true,

  locales: {
    zh: {
      label: '简体中文',
      lang: 'zh-CN',
      link: '/zh/',
      themeConfig: {
        nav: [
          { text: '首页', link: '/zh/' },
          { text: '快速开始', link: '/zh/guide/getting-started' },
          { text: '课程模块', link: '/zh/modules/' },
          { text: 'GitHub', link: 'https://github.com/zimingttkx/AI-Practices' }
        ],
        sidebar: {
          '/zh/guide/': [
            {
              text: '入门指南',
              items: [
                { text: '快速开始', link: '/zh/guide/getting-started' },
                { text: '安装配置', link: '/zh/guide/installation' },
                { text: '项目结构', link: '/zh/guide/architecture' }
              ]
            }
          ],
          '/zh/modules/': [
            {
              text: '学习模块',
              items: [
                { text: '模块概览', link: '/zh/modules/' },
                { text: '01 - 机器学习基础', link: '/zh/modules/01-foundations' },
                { text: '02 - 神经网络', link: '/zh/modules/02-neural-networks' },
                { text: '03 - 计算机视觉', link: '/zh/modules/03-computer-vision' },
                { text: '04 - 序列模型', link: '/zh/modules/04-sequence-models' },
                { text: '05 - 高级专题', link: '/zh/modules/05-advanced' },
                { text: '06 - 生成模型', link: '/zh/modules/06-generative' },
                { text: '07 - 强化学习', link: '/zh/modules/07-reinforcement-learning' },
                { text: '08 - 理论笔记', link: '/zh/modules/08-theory' },
                { text: '09 - 实战项目', link: '/zh/modules/09-projects' }
              ]
            }
          ]
        },
        outline: {
          label: '页面导航'
        },
        docFooter: {
          prev: '上一页',
          next: '下一页'
        },
        lastUpdated: {
          text: '最后更新于'
        },
        editLink: {
          pattern: 'https://github.com/zimingttkx/AI-Practices/edit/main/docs/:path',
          text: '在 GitHub 上编辑此页'
        }
      }
    },
    en: {
      label: 'English',
      lang: 'en-US',
      link: '/en/',
      themeConfig: {
        nav: [
          { text: 'Home', link: '/en/' },
          { text: 'Quick Start', link: '/en/guide/getting-started' },
          { text: 'Modules', link: '/en/modules/' },
          { text: 'GitHub', link: 'https://github.com/zimingttkx/AI-Practices' }
        ],
        sidebar: {
          '/en/guide/': [
            {
              text: 'Guide',
              items: [
                { text: 'Getting Started', link: '/en/guide/getting-started' },
                { text: 'Installation', link: '/en/guide/installation' },
                { text: 'Architecture', link: '/en/guide/architecture' }
              ]
            }
          ],
          '/en/modules/': [
            {
              text: 'Learning Modules',
              items: [
                { text: 'Overview', link: '/en/modules/' },
                { text: '01 - ML Foundations', link: '/en/modules/01-foundations' },
                { text: '02 - Neural Networks', link: '/en/modules/02-neural-networks' },
                { text: '03 - Computer Vision', link: '/en/modules/03-computer-vision' },
                { text: '04 - Sequence Models', link: '/en/modules/04-sequence-models' },
                { text: '05 - Advanced Topics', link: '/en/modules/05-advanced' },
                { text: '06 - Generative Models', link: '/en/modules/06-generative' },
                { text: '07 - Reinforcement Learning', link: '/en/modules/07-reinforcement-learning' },
                { text: '08 - Theory Notes', link: '/en/modules/08-theory' },
                { text: '09 - Projects', link: '/en/modules/09-projects' }
              ]
            }
          ]
        },
        outline: {
          label: 'On this page'
        },
        editLink: {
          pattern: 'https://github.com/zimingttkx/AI-Practices/edit/main/docs/:path',
          text: 'Edit this page on GitHub'
        }
      }
    }
  },

  themeConfig: {
    logo: '/logo.svg',

    socialLinks: [
      { icon: 'github', link: 'https://github.com/zimingttkx/AI-Practices' }
    ],

    search: {
      provider: 'local',
      options: {
        locales: {
          zh: {
            translations: {
              button: {
                buttonText: '搜索文档',
                buttonAriaLabel: '搜索文档'
              },
              modal: {
                noResultsText: '无法找到相关结果',
                resetButtonTitle: '清除查询条件',
                footer: {
                  selectText: '选择',
                  navigateText: '切换'
                }
              }
            }
          }
        }
      }
    },

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024-present AI-Practices'
    }
  }
})
