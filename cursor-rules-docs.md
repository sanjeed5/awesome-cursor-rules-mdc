export const meta = {
  title: "Rules",
  description: "Configure persistent instructions with Project, Team, and User Rules, plus AGENTS.md. Learn best practices for effective coding guidelines."
};

# Rules

Rules provide system-level instructions to Agent. They bundle prompts, scripts, and more together, making it easy to manage and share workflows across your team.

Cursor supports four types of rules:

<CardGroup cols={2}>
  <Card title="Project Rules" icon="folder-tree">
    Stored in `.cursor/rules`, version-controlled and scoped to your codebase.
  </Card>
  <Card title="User Rules" icon="user">
    Global to your Cursor environment. Used by Agent (Chat).
  </Card>
  <Card title="Team Rules" icon="users">
    Team-wide rules managed from the dashboard. Available on Team and [Enterprise](/docs/enterprise) plans.
  </Card>
  <Card title="AGENTS.md" icon="bot">
    Agent instructions in markdown format. Simple alternative to
    `.cursor/rules`.
  </Card>
</CardGroup>

## How rules work

Large language models don't retain memory between completions. Rules provide persistent, reusable context at the prompt level.

When applied, rule contents are included at the start of the model context. This gives the AI consistent guidance for generating code, interpreting edits, or helping with workflows.

<Image
  height={674}
  width={1198}
  lightSrc="/docs-static/images/context/rules/rules-applied.png"
  alt="Rule applied in context with chat"
/>



## Project rules

Project rules live in `.cursor/rules`. Each rule is a folder containing a `RULE.md` file and is version-controlled. They are scoped using path patterns, invoked manually, or included based on relevance.

Use project rules to:

- Encode domain-specific knowledge about your codebase
- Automate project-specific workflows or templates
- Standardize style or architecture decisions

### Rule folder structure

Each rule folder can contain:

- **`RULE.md`** — The main rule file with frontmatter metadata and instructions
- **Scripts and prompts** — Additional files referenced by the rule

```bash
.cursor/rules/
  my-rule/
    RULE.md           # Main rule file
    scripts/          # Helper scripts (optional)
```

### Rule anatomy

Each rule is a folder containing a `RULE.md` file with frontmatter metadata and content. Control how rules are applied from the type dropdown which changes properties `description`, `globs`, `alwaysApply`.

| <span className="no-wrap">Rule Type</span>                    | Description                                                                      |
| :--------------------------------------------------------- | :------------------------------------------------------------------------------- |
| <span className="no-wrap">`Always Apply`</span>             | Apply to every chat session                                                     |
| <span className="no-wrap">`Apply Intelligently`</span>      | When Agent decides it's relevant based on description                           |
| <span className="no-wrap">`Apply to Specific Files`</span> | When file matches a specified pattern                                           |
| <span className="no-wrap">`Apply Manually`</span>           | When @-mentioned in chat (e.g., `@my-rule`)                                     |

```md
---
globs:
alwaysApply: false
---

- Use our internal RPC pattern when defining services
- Always use snake_case for service names.

@service-template.ts
```

### Creating a rule

Create rules using the `New Cursor Rule` command or going to `Cursor Settings > Rules, Commands`. This creates a new rule folder in `.cursor/rules`. From settings you can see all rules and their status.

<Image
  height={3334}
  width={6016}
  lightSrc="/docs-static/images/context/rules/rule-settings.png"
  alt="Comparison of concise vs long rules"
/>

## Best practices

Good rules are focused, actionable, and scoped.

- Keep rules under 500 lines
- Split large rules into multiple, composable rules
- Provide concrete examples or referenced files
- Avoid vague guidance. Write rules like clear internal docs
- Reuse rules when repeating prompts in chat

## RULE.md file format

The `RULE.md` file is a markdown file with frontmatter metadata and content. The frontmatter metadata is used to control how the rule is applied. The content is the rule itself.

```markdown
---
description: "This rule provides standards for frontend components and API validation"
alwaysApply: false
---

...rest of the rule content
```

If alwaysApply is true, the rule will be applied to every chat session. Otherwise, the description of the rule will be presented to the Cursor Agent to decide if it should be applied.

## Examples

<AccordionGroup>
  <Accordion title="Standards for frontend components and API validation">
    This rule provides standards for frontend components:

    When working in components directory:
    - Always use Tailwind for styling
    - Use Framer Motion for animations
    - Follow component naming conventions

    This rule enforces validation for API endpoints:

    In API directory:
    - Use zod for all validation
    - Define return types with zod schemas
    - Export types generated from schemas

  </Accordion>

  <Accordion title="Templates for Express services and React components">
    This rule provides a template for Express services:

    Use this template when creating Express service:
    - Follow RESTful principles
    - Include error handling middleware
    - Set up proper logging

    @express-service-template.ts

    This rule defines React component structure:

    React components should follow this layout:
    - Props interface at top
    - Component as named export
    - Styles at bottom

    @component-template.tsx

  </Accordion>

  <Accordion title="Automating development workflows and documentation generation">
    This rule automates app analysis:

    When asked to analyze the app:
    1. Run dev server with `npm run dev`
    2. Fetch logs from console
    3. Suggest performance improvements

    This rule helps generate documentation:

    Help draft documentation by:
    - Extracting code comments
    - Analyzing README.md
    - Generating markdown documentation

  </Accordion>

  <Accordion title="Adding a new setting in Cursor" >
First create a property to toggle in `@reactiveStorageTypes.ts`.

Add default value in `INIT_APPLICATION_USER_PERSISTENT_STORAGE` in `@reactiveStorageService.tsx`.

For beta features, add toggle in `@settingsBetaTab.tsx`, otherwise add in `@settingsGeneralTab.tsx`. Toggles can be added as `<SettingsSubSection>` for general checkboxes. Look at the rest of the file for examples.

```jsx
<SettingsSubSection
  label="Your feature name"
  description="Your feature description"
  value={
    vsContext.reactiveStorageService.applicationUserPersistentStorage
      .myNewProperty ?? false
  }
  onChange={(newVal) => {
    vsContext.reactiveStorageService.setApplicationUserPersistentStorage(
      "myNewProperty",
      newVal,
    );
  }}
/>
```

To use in the app, import reactiveStorageService and use the property:

```js
const flagIsEnabled =
  vsContext.reactiveStorageService.applicationUserPersistentStorage
    .myNewProperty;
```

  </Accordion>

</AccordionGroup>

<br />

Examples are available from providers and frameworks. Community-contributed rules are found across crowdsourced collections and repositories online.

## Team Rules

Team and [Enterprise](/docs/enterprise) plans can create and enforce rules across their entire organization from the [Cursor dashboard](https://cursor.com/dashboard?tab=team-content). Admins can configure whether or not each rule is required for team members.

Team Rules work alongside other rule types and take precedence to ensure organizational standards are maintained across all projects. They provide a powerful way to ensure consistent coding standards, practices, and workflows across your entire team without requiring individual setup or configuration.

### Managing Team Rules

Team administrators can create and manage rules directly from the Cursor dashboard:

<Image
  height={600}
  width={1000}
  lightSrc="/docs-static/images/context/rules/team-rules-empty.png"
  alt="Empty team rules dashboard where team administrators can add new rules"
/>

Once team rules are created, they automatically apply to all team members and are visible in the dashboard:

<Image
  height={600}
  width={1000}
  lightSrc="/docs-static/images/context/rules/team-rules-1.png"
  alt="Team rules dashboard showing a single team rule that will be enforced for all team members"
/>

### Activation and enforcement

- **Enable this rule immediately**: When checked, the rule is active as soon as you create it. When unchecked, the rule is saved as a draft and does not apply until you enable it later.
- **Enforce this rule**: When enabled, the rule is required for all team members and cannot be disabled in their Cursor settings. When not enforced, team members can toggle the rule off in `Cursor Settings → Rules` under the Team Rules section.

<Info>
By default, non‑enforced Team Rules can be disabled by users. Use <strong>Enforce this rule</strong> to prevent that.
</Info>

### Format and how Team Rules are applied

- **Plain text**: Team Rules are free‑form text. They do not use the folder structure of Project Rules and do not support metadata such as `globs`, `alwaysApply`, or rule types.
- **Where they apply**: When a Team Rule is enabled (and not disabled by the user, unless enforced), it is included in the model context for Agent (Chat) across all repositories and projects for that team.
- **Precedence**: Rules are applied in this order: **Team Rules → Project Rules → User Rules**. All applicable rules are merged; earlier sources take precedence when guidance conflicts.

<Info>
Some teams use enforced rules as part of internal compliance workflows. While this is supported, AI guidance should not be your only security control.
</Info>

## Importing Rules

You can import rules from external sources to reuse existing configurations or bring in rules from other tools.

### Remote rules (via GitHub)

Import rules directly from any GitHub repository you have access to—public or private.

1. Open **Cursor Settings → Rules, Commands**
2. Click `+ Add Rule` next to `Project Rules`, then select Remote Rule (Github)
3. Paste the GitHub repository URL containing the rule
4. Cursor will pull and sync the rule into your project

Imported rules stay synced with their source repository, so updates to the remote rule are automatically reflected in your project.

### Claude skills and plugins

Cursor can load rules from Claude's skills and plugins system. These imported rules are always applied as agent-decided rules, meaning Cursor determines when they are relevant based on context.

To enable or disable Claude skills and plugins:

1. Open **Cursor Settings → Rules**
2. Find the **Import Settings** section
3. Toggle **Claude skills and plugins** on or off

<Info>
Claude skills and plugins are treated as agent-decided rules and cannot be configured as always-apply or manual rules.
</Info>

## AGENTS.md

`AGENTS.md` is a simple markdown file for defining agent instructions. Place it in your project root as an alternative to `.cursor/rules` for straightforward use cases.

Unlike Project Rules, `AGENTS.md` is a plain markdown file without metadata or complex configurations. It's perfect for projects that need simple, readable instructions without the overhead of structured rules.

Cursor supports AGENTS.md in the project root and subdirectories.

```markdown
# Project Instructions

## Code Style

- Use TypeScript for all new files
- Prefer functional components in React
- Use snake_case for database columns

## Architecture

- Follow the repository pattern
- Keep business logic in service layers
```

### Improvements

<AccordionGroup>
  <Accordion title="Nested AGENTS.md support">
    Nested `AGENTS.md` support in subdirectories is now available. You can place `AGENTS.md` files in any subdirectory of your project, and they will be automatically applied when working with files in that directory or its children.

    This allows for more granular control of agent instructions based on the area of your codebase you're working in:

    ```bash
    project/
      AGENTS.md              # Global instructions
      frontend/
        AGENTS.md            # Frontend-specific instructions
        components/
          AGENTS.md          # Component-specific instructions
      backend/
        AGENTS.md            # Backend-specific instructions
    ```

    Instructions from nested `AGENTS.md` files are combined with parent directories, with more specific instructions taking precedence.
  </Accordion>
</AccordionGroup>

## User Rules

User Rules are global preferences defined in **Cursor Settings → Rules** that apply across all projects. They are used by Agent (Chat) and are perfect for setting preferred communication style or coding conventions:

```md
Please reply in a concise style. Avoid unnecessary repetition or filler language.
```

## Legacy Cursor Rules

### `.cursorrules`

The `.cursorrules` (legacy) file in your project root is still supported but **will be deprecated**. We recommend migrating to Project Rules or to `AGENTS.md`.

### `.mdc` cursor rules

As of 2.2, `.mdc` cursor rules will remain functional however all new rules will now be created as folders in `.cursor/rules`. This is to improve the readability and maintainability of rules.

## FAQ

<AccordionGroup>
  <Accordion title="Why isn't my rule being applied?">
    Check the rule type. For `Apply Intelligently`, ensure a description is defined. For `Apply to Specific Files`, ensure the file pattern matches referenced files.
  </Accordion>

<Accordion title="Can rules reference other rules or files?">
  Yes. Use `@filename.ts` to include files in your rule's context. You can also @mention rules in chat to apply them manually.
</Accordion>

<Accordion title="Can I create a rule from chat?">
  Yes, you can ask the agent to create a new rule for you.
</Accordion>

  <Accordion title="Do rules impact Cursor Tab or other AI features?">
    No. Rules do not impact Cursor Tab or other AI features.
  </Accordion>

  <Accordion title="Do User Rules apply to Inline Edit (Cmd/Ctrl+K)?">
    No. User Rules are not applied to Inline Edit (Cmd/Ctrl+K). They are only
    used by Agent (Chat).
  </Accordion>
</AccordionGroup>
