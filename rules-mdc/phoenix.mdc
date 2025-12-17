---
description: Provides definitive guidelines for building robust, maintainable, and performant Phoenix applications using modern Elixir best practices, focusing on context-driven design, strict formatting, and effective testing.
globs: **/*
---
# phoenix Best Practices

This guide outlines the definitive best practices for developing Phoenix applications. Adhere to these rules to ensure consistency, maintainability, and high performance across our codebase.

## Code Organization and Structure

### 1. Context-Driven Design
Encapsulate all data access and business logic for a feature within a dedicated context module. Controllers and LiveViews must remain thin, focusing solely on HTTP/WebSocket handling.

**✅ GOOD:**
```elixir
# lib/my_app/accounts.ex
defmodule MyApp.Accounts do
  alias MyApp.Repo
  alias MyApp.Accounts.User

  def create_user(attrs) do
    %User{}
    |> User.changeset(attrs)
    |> Repo.insert()
  end

  def get_user!(id), do: Repo.get!(User, id)
end

# lib/my_app_web/controllers/user_controller.ex
defmodule MyAppWeb.UserController do
  use MyAppWeb, :controller
  alias MyApp.Accounts

  def create(conn, %{"user" => user_params}) do
    case Accounts.create_user(user_params) do
      {:ok, user} ->
        conn
        |> put_flash(:info, "User created successfully.")
        |> redirect(to: ~p"/users/#{user}")
      {:error, %Ecto.Changeset{} = changeset} ->
        conn
        |> put_flash(:error, "Failed to create user.")
        |> render("new.html", changeset: changeset)
    end
  end
end
```

### 2. Single Module Per Source File
Each `.ex` file must contain a single module. This includes structs.

**❌ BAD:**
```elixir
# lib/payment/worker.ex
defmodule Payment.Worker do
  defmodule WorkerConfiguration do
    defstruct [:retry_period_ms, :error_retry_limit]
  end
  # ...
end
```

**✅ GOOD:**
```elixir
# lib/payment/worker.ex
defmodule Payment.Worker do
  # ...
end

# lib/payment/worker_configuration.ex
defmodule Payment.WorkerConfiguration do
  defstruct [:retry_period_ms, :error_retry_limit]
end
```

### 3. Naming Conventions
Adhere strictly to Elixir's naming conventions.

*   **Modules:** `PascalCase`
*   **Functions, Variables, Files, Directories:** `snake_case`

**❌ BAD:**
```elixir
defmodule Payment.jobs.Inquiry_scheduler do # Module, file, dir naming
  defp calculateBalance(%Entry{bookType: "debit", amount: amount}, balance) do # Function, variable naming
    balance - amount
  end
end
```

**✅ GOOD:**
```elixir
# lib/payment/jobs/inquiry_scheduler.ex
defmodule Payment.Jobs.InquiryScheduler do
  defp calculate_balance(%Entry{book_type: "debit", amount: amount}, balance) do
    balance - amount
  end
end
```

### 4. Code Formatting
Always run `mix format`. Configure `.formatter.exs` with `line_length: 100`. Use 2-space soft tabs. End every file with a newline.

## Common Patterns and Anti-patterns

### 1. Bang Functions (`!`) for Raising Errors
Suffix functions that are expected to raise an exception (e.g., on invalid input or missing record) with `!`. This clearly signals to callers that the function might fail.

**✅ GOOD:**
```elixir
def create_user!(registration_uid, customer_params) do
  # ... logic that might raise ArgumentError or Ecto.NoResultsError
end

defp build_attrs!(%{
  "alien_id" => alien_id,
  "citizen_id" => citizen_id,
}) do
  %{citizen_id: citizen_id, alien_id: alien_id}
end

defp build_attrs!(_params) do
  raise ArgumentError, message: "Unexpected customer profile params"
end
```

### 2. Prefer Pattern Matching and Guards
Use pattern matching and guards in function definitions over `if/else` statements within the function body for clearer, more declarative logic.

**❌ BAD:**
```elixir
defp handle_response({:ok, %Response{status: status, body: body}}) do
  if status in 200..299 do
    {:ok, body}
  else
    {:error, :internal_server_error, reason}
  end
end
```

**✅ GOOD:**
```elixir
defp handle_response({:ok, %Response{status: status, body: body}})
     when status in 200..299,
     do: {:ok, body}

defp handle_response({:error, reason}),
     do: {:error, :internal_server_error, reason}
```

### 3. Pipe Operator Usage
Avoid using the pipe operator for a single function call. Start a pipeline with the raw value. Use `Kernel.then/2` for anonymous functions in a pipeline.

**❌ BAD:**
```elixir
some_string |> String.downcase()
value |> (&(&1 * 10)).()
```

**✅ GOOD:**
```elixir
String.downcase(some_string)
some_string |> String.trim() |> String.downcase() # Raw value start
value |> then(&(&1 * 10)) # then/2 for anonymous functions
```

### 4. Alias Grouping
Use shorthand syntax to group aliases from the same submodule.

**❌ BAD:**
```elixir
alias Payment.Ledger.Balance
alias Payment.Ledger.EntriesGenerator
```

**✅ GOOD:**
```elixir
alias Payment.Ledger.{Balance, EntriesGenerator}
```

### 5. Conditional Statements
Use `do:` for single-line `if/unless`. Never use `unless` with `else`; rewrite with `if`.

**❌ BAD:**
```elixir
if some_condition do "some_stuff" end
unless success do IO.puts("failure") else IO.puts("success") end
```

**✅ GOOD:**
```elixir
if some_condition, do: "some_stuff"
if success, do: IO.puts("success"), else: IO.puts("failure")
```

### 6. Ignored Variables
Use `_variable_name` for ignored variables to improve readability.

**❌ BAD:**
```elixir
def index(conn, _), do: Plug.Conn.resp(conn, :ok, "success")
```

**✅ GOOD:**
```elixir
def index(conn, _params), do: Plug.Conn.resp(conn, :ok, "success")
```

### 7. Private Functions
Place all private functions at the bottom of the module.

## Performance Considerations

Phoenix leverages the Erlang VM for high concurrency and fault tolerance.
*   **Ecto Queries**: Optimize database queries. Use `Ecto.Query.preload/3` to avoid N+1 issues.
*   **LiveView**: Embrace LiveView for interactive UIs to minimize client-side JavaScript and network roundtrips.
*   **Concurrency**: Design long-running tasks to run in separate processes (e.g., using `Task` or `GenServer`) to avoid blocking the request cycle.

## Common Pitfalls and Gotchas

### 1. Scattering Business Logic
The most common pitfall is scattering business logic across controllers, LiveViews, and schemas. Always centralize it in contexts.

### 2. Over-reliance on Metaprogramming
While powerful, excessive `use` or `import` can obscure where functions originate. Rely on documentation and `h/1` in IEx to understand module APIs.

### 3. Deployment Complexity
Elixir Releases with Docker simplify deployment, but still require careful configuration of environment variables and build steps (e.g., `npm install` for assets). Use `mix release` and the generated `Dockerfile`.

## Security Best Practices

### 1. Authentication
Use `mix phx.gen.auth` for a robust, production-ready authentication system.

### 2. Input Validation
Always validate user input using Ecto Changesets. Never trust client-side data.

**✅ GOOD:**
```elixir
def changeset(user, attrs) do
  user
  |> cast(attrs, [:email, :password, :password_confirmation])
  |> validate_required([:email, :password])
  |> validate_length(:password, min: 8)
  |> unique_constraint(:email)
  |> put_assoc(:profile, profile_changeset(user.profile, attrs))
end
```

### 3. XSS and CSRF Protection
Phoenix provides built-in protection against XSS (via HTML escaping in HEEx templates) and CSRF (via `Plug.CSRFProtection`). Ensure these are active.

## Error Handling

### 1. Return Tuples for Expected Outcomes
For operations that can succeed or fail, consistently return `{:ok, result}` or `{:error, reason}` tuples.

**✅ GOOD:**
```elixir
def update_user(user, attrs) do
  user
  |> User.changeset(attrs)
  |> Repo.update()
end
```

### 2. Centralized Logging and Error Tracking
Integrate with a centralized logging solution (e.g., Graylog via `GelfLogger`) and an error tracking service (e.g., Sentry) for production monitoring.

## API Design

### 1. Contexts as API Boundaries
Design your contexts to expose a clear, stable API for interacting with your application's data and business logic.

### 2. RESTful JSON APIs
Use `mix phx.gen.json` to scaffold RESTful JSON APIs. Ensure consistent response formats and HTTP status codes.

## Testing Approaches

### 1. Comprehensive ExUnit Testing
Write tests for all layers: contexts, controllers, LiveViews, and channels. Run tests with `mix test`.

### 2. Leverage Generated Test Cases
Phoenix generators (`mix phx.gen.live`, `mix phx.gen.html`, etc.) provide excellent examples for testing each component.

### 3. `ConnCase` and `DataCase`
Use `HelloWeb.ConnCase` for controller/LiveView tests (provides `conn` and web helpers) and `Hello.DataCase` for context/Ecto tests (manages database sandbox).

**✅ GOOD:**
```elixir
# test/my_app_web/controllers/user_controller_test.exs
defmodule MyAppWeb.UserControllerTest do
  use MyAppWeb.ConnCase, async: true
  alias MyApp.Accounts

  setup [:create_user]

  defp create_user(_) do
    {:ok, user} = Accounts.create_user(%{email: "test@example.com", password: "password"})
    %{user: user}
  end

  test "shows existing user", %{conn: conn, user: user} do
    conn = get(conn, ~p"/users/#{user}")
    assert html_response(conn, 200) =~ user.email
  end
end
```

### 4. LiveView Testing Helpers
For LiveView tests, use `render_hook/2`, `render_component/2`, `form/2`, `assert_patch/2`, etc. When debugging HTML output, use specific element selectors rather than broad string matching.

**✅ GOOD (using custom helpers for clarity):**
```elixir
# test/support/element_helpers.ex
defmodule MyApp.Support.ElementHelpers do
  import Phoenix.LiveViewTest

  def get_by_role(view, :button, text_filter) do
    button = view |> element("button", text_filter)
    assert has_element?(button)
    button
  end

  def get_by_role(view, :textbox, text_filter) do
    label = view |> element("label", text_filter)
    textbox_id = Regex.run(~r/(?<=for=")\w+(?=")/, label |> render) |> Enum.at(0)
    input = view |> element("input##{textbox_id}")
    textbox = if has_element?(input), do: input, else: view |> element("textarea##{textbox_id}")
    assert has_element?(label)
    assert has_element?(textbox)
    textbox
  end
end

# test/my_app_web/live/user_live_test.exs
defmodule MyAppWeb.UserLiveTest do
  use MyAppWeb.ConnCase, async: true
  import Phoenix.LiveViewTest
  import MyApp.Support.ElementHelpers # Import your custom helpers

  test "creates a new user", %{conn: conn} do
    {:ok, lv, _html} = live(conn, ~p"/users/new")

    lv
    |> get_by_role(:textbox, "Email")
    |> render_change(value: "new@example.com")

    lv
    |> get_by_role(:textbox, "Password")
    |> render_change(value: "securepassword")

    lv
    |> get_by_role(:button, "Create User")
    |> render_click()

    assert_patch(lv, ~p"/users")
    assert has_element?(lv.html, "User created successfully.")
  end
end
```