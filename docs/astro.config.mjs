// @ts-check
import { defineConfig } from 'astro/config';
import path from "node:path";
import starlight from '@astrojs/starlight';
import starlightLinksValidator from 'starlight-links-validator';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import starlightThemeFlexoki from 'starlight-theme-flexoki'

// https://astro.build/config
export default defineConfig({
	site: 'https://neilkichler.github.io',
	base: '/cuinterval',
	prefetch: {
		prefetchAll: true
	},
	vite: {
		resolve: {
			alias: {
				"@assets": path.resolve("./src/assets"),
				"@components": path.resolve("./src/components"),
			}
		},
	},
	markdown: {
		remarkPlugins: [remarkMath],
		rehypePlugins: [rehypeKatex],
	},
	integrations: [
		starlight({
			favicon: '/favicon.png',
			title: 'CuInterval',
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/neilkichler/cuinterval' },
			],
			customCss: ['katex/dist/katex.min.css'],
			sidebar: [
				{
					label: 'Guides',
					autogenerate: { directory: 'guides' },
				},
				{
					label: 'Operations',
					autogenerate: { directory: 'operations' },
					collapsed: true
				},
				{
					label: 'Reference',
					autogenerate: { directory: 'reference' },
				},
				{
					label: 'Resources',
					autogenerate: { directory: 'resources' },
				},
			],
			plugins: [
				starlightLinksValidator({
					errorOnRelativeLinks: false,
				}),
				starlightThemeFlexoki()
			]
		}),
	],
});
