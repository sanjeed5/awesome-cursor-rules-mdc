---
description: Definitive guidelines for configuring and using Vim/Neovim as a high-performance, terminal-centric IDE, emphasizing modern practices, efficient plugin management, and advanced workflows.
globs: **/*
---
# vim Best Practices

Vim (and especially Neovim) is a powerful, programmable editor. Treat it as a foundation for a custom IDE. This guide focuses on modern Neovim practices, which are largely applicable to Vim 8+ with appropriate plugin choices.

## 1. Configuration Organization & Structure

**Always use a structured configuration.** For Neovim, this means Lua files in `~/.config/nvim/`. For Vim, use `~/.vimrc`. Lua offers superior expressiveness and performance.

**✅ GOOD: Neovim with Lua (recommended)**
Organize your `init.lua` by sourcing smaller, focused files.

```lua
-- ~/.config/nvim/init.lua
require('options')    -- Core editor settings
require('keymaps')     -- Custom keybindings
require('plugins')    -- Plugin manager setup and lazy loading
require('autocmds')   -- Automatic commands
```

**❌ BAD: Monolithic `init.vim` / `init.lua`**
A single, unorganized file becomes unmanageable and hard to debug.

```vim
" ~/.vimrc (or init.lua) - Avoid this sprawl
set number
map <leader>s :w<CR>
" ... hundreds of lines ...
```

## 2. Essential Configuration Patterns

### 2.1. Sensible Defaults

Disable `compatible` mode immediately. Set fundamental options for a modern editing experience.

**✅ GOOD: Modern Defaults**

```lua
-- In lua/options.lua
vim.opt.termguicolors = true   -- Enable true colors in terminal
vim.opt.number = true          -- Show line numbers
vim.opt.relativenumber = true  -- Show relative line numbers
vim.opt.expandtab = true       -- Use spaces instead of tabs
vim.opt.shiftwidth = 4         -- Number of spaces for indent
vim.opt.tabstop = 4            -- Number of spaces a tab counts for
vim.opt.autoindent = true      -- Copy indent from previous line
vim.opt.smartindent = true     -- Smart auto-indent for C-like files
vim.opt.undofile = true        -- Persistent undo history
vim.opt.backup = false         -- No backup files (use Git!)
vim.opt.writebackup = false    -- No backup files during write
vim.opt.swapfile = false       -- No swap files
vim.opt.hlsearch = true        -- Highlight search results
vim.opt.incsearch = true       -- Incremental search
vim.opt.scrolloff = 8          -- Keep 8 lines of context
vim.opt.updatetime = 300       -- Faster UI updates for plugins
vim.opt.timeoutlen = 500       -- Shorter key sequence timeout
vim.opt.cmdheight = 1          -- Command line height
vim.opt.laststatus = 2         -- Always show status line
vim.opt.showmode = false       -- Don't show -- INSERT -- etc.
vim.opt.signcolumn = 'yes'     -- Always show sign column
vim.opt.cursorline = true      -- Highlight current line
vim.opt.wrap = false           -- No line wrapping
vim.opt.mouse = 'a'            -- Enable mouse in all modes
vim.opt.clipboard = 'unnamedplus' -- Sync with system clipboard
```

**❌ BAD: Vi-compatible or minimal defaults**
Sticking to Vi's ancient defaults hinders productivity.

```vim
" ~/.vimrc
set compatible " NEVER do this!
set nobackup   " This is fine, but missing many other essentials
```

### 2.2. Key Mappings

Use `<leader>` for all custom mappings to avoid conflicts and provide a consistent prefix.

**✅ GOOD: Leader-prefixed Mappings**

```lua
-- In lua/keymaps.lua
vim.g.mapleader = ' ' -- Set space as the leader key

-- Save file
vim.keymap.set('n', '<leader>w', '<cmd>w<CR>', { desc = 'Save File' })
-- Toggle NvimTree
vim.keymap.set('n', '<leader>e', '<cmd>NvimTreeToggle<CR>', { desc = 'Toggle File Explorer' })
```

**❌ BAD: Overwriting built-in commands**
Directly mapping common keys like `s` or `q` leads to frustration.

```vim
" ~/.vimrc
map s :w<CR> " This overwrites the 'substitute' command!
```

### 2.3. Asynchronous Plugin Management

**Always use a lazy-loading plugin manager.** `lazy.nvim` is the current standard for Neovim. It drastically improves startup time by loading plugins only when needed.

**✅ GOOD: Lazy Loading with `lazy.nvim`**

```lua
-- In lua/plugins.lua
local lazypath = vim.fn.stdpath("data") .. "/lazy/lazy.nvim"
if not vim.loop.fs_stat(lazypath) then
  vim.fn.system({
    "git", "clone", "--filter=blob:none",
    "https://github.com/folke/lazy.nvim.git", lazypath
  })
end
vim.opt.rtp:prepend(lazypath)

require("lazy").setup({
  -- LSP Configuration
  {
    'neovim/nvim-lspconfig',
    dependencies = {
      'williamboman/mason.nvim',
      'williamboman/mason-lspconfig.nvim',
    },
    config = function()
      require('lsp_config') -- Separate LSP setup
    end
  },
  -- Fuzzy Finder
  {
    'nvim-telescope/telescope.nvim',
    tag = '0.1.x',
    dependencies = { 'nvim-lua/plenary.nvim' },
    keys = {
      { '<leader>ff', '<cmd>Telescope find_files<CR>', desc = 'Find Files' },
      { '<leader>fg', '<cmd>Telescope live_grep<CR>', desc = 'Live Grep' },
    },
  },
  -- Tree-sitter for syntax highlighting
  {
    'nvim-treesitter/nvim-treesitter',
    build = ':TSUpdate',
    config = function()
      require('nvim-treesitter.configs').setup {
        ensure_installed = { "c", "cpp", "go", "lua", "python", "rust", "typescript" },
        highlight = { enable = true },
        indent = { enable = true },
      }
    end,
  },
  -- Status line
  {
    'nvim-lualine/lualine.nvim',
    config = function() require('lualine').setup() end,
  },
  -- File explorer
  {
    'nvim-tree/nvim-tree.lua',
    lazy = true, -- Only load when NvimTreeToggle is called
    cmd = { 'NvimTreeToggle', 'NvimTreeOpen' },
    config = function() require('nvim-tree').setup() end,
  },
})
```

**❌ BAD: Synchronous plugin loading**
Loading all plugins at startup, especially heavy ones, makes Vim slow.

```vim
" ~/.vimrc
packadd nvim-lspconfig " Loads immediately, even if not needed
```

## 3. Performance Considerations

Prioritize asynchronous operations and minimize UI redraws.

**✅ GOOD: Asynchronous Linting & Formatting**
Use Neovim's built-in LSP diagnostics for real-time feedback. Format on save via `autocmd`.

```lua
-- In lua/lsp_config.lua (example for formatting)
vim.api.nvim_create_autocmd('BufWritePre', {
  pattern = '*',
  callback = function(args)
    vim.lsp.buf.format({ async = true, bufnr = args.buf })
  end,
})
```

**❌ BAD: Blocking external commands**
Running `!fmt %` blocks the editor until the command finishes.

```vim
" ~/.vimrc
autocmd BufWritePre *.go !goimports -w % " Blocks Vim
```

## 4. Common Pitfalls & Gotchas

### 4.1. Forgetting `set nocompatible`

This is the most common mistake for new Vim users. Always disable `compatible` mode.

**❌ BAD:**
```vim
" ~/.vimrc
" (no set nocompatible) -> Vim behaves like ancient Vi
```
**✅ GOOD:**
```lua
-- In lua/options.lua
vim.opt.compatible = false
```

### 4.2. Ignoring LSP

Modern IDE features (go-to-definition, refactoring, autocompletion) are powered by Language Servers. Integrate `nvim-lspconfig` with `mason.nvim`.

**❌ BAD: Relying solely on regex-based tools**
`grep` and `ctags` are useful but lack semantic understanding.

```vim
" ~/.vimrc
map <leader>gd :grep -R <cword> .<CR> " Not semantic
```
**✅ GOOD: Full LSP Integration**

```lua
-- In lua/lsp_config.lua
local lspconfig = require('lspconfig')
local mason_lspconfig = require('mason-lspconfig')
local capabilities = vim.lsp.protocol.make_client_capabilities()
-- ... add completion capabilities etc.

mason_lspconfig.setup_handlers {
  function(server_name)
    lspconfig[server_name].setup {
      capabilities = capabilities,
      -- ... other server-specific settings
    }
  end,
}
```

## 5. Testing & Debugging Approaches

Leverage Neovim's built-in terminal and quickfix list for efficient testing and error navigation.

**✅ GOOD: Integrated Terminal for Tests**
Run tests in a split window and navigate output.

```lua
-- In lua/keymaps.lua
vim.keymap.set('n', '<leader>tt', '<cmd>split term://go test ./...<CR>', { desc = 'Run Go Tests' })
vim.keymap.set('n', '<leader>tc', '<cmd>split term://npm test<CR>', { desc = 'Run NPM Tests' })
```

**❌ BAD: Switching to external terminal**
Context switching breaks flow and is inefficient.

```vim
" ~/.vimrc
map <leader>t :!go test ./...<CR> " Blocks Vim, output not easily navigable
```

**✅ GOOD: Quickfix List for Errors**
Populate the quickfix list with linter/compiler errors and use `:cprev` / `:cnext` to navigate.

```lua
-- LSP diagnostics automatically populate the quickfix/location list.
-- Ensure you have mappings for navigation:
vim.keymap.set('n', '[d', vim.diagnostic.goto_prev, { desc = 'Go to previous diagnostic' })
vim.keymap.set('n', ']d', vim.diagnostic.goto_next, { desc = 'Go to next diagnostic' })
vim.keymap.set('n', '<leader>qf', '<cmd>copen<CR>', { desc = 'Open Quickfix List' })
```