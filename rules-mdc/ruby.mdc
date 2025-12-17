---
description: This guide provides opinionated, actionable best practices for writing modern Ruby code, focusing on maintainability, performance, and security, with a strong emphasis on RuboCop and Rails conventions.
globs: **/*
---
# Ruby Best Practices

Our team adheres to a strict set of Ruby best practices, primarily enforced by RuboCop and aligned with the official Ruby and Rails Style Guides. This document serves as your definitive reference for writing clean, efficient, and maintainable Ruby code.

## 1. Code Organization & Structure

Adhere to the Single Responsibility Principle (SRP) and keep files focused.

### 1.1 Single Class/Module Per File
Each Ruby file must define a single class or module. Name the file using `snake_case` corresponding to the `CamelCase` class/module name.

❌ BAD
```ruby
# user.rb
class User; end
class UserProfile; end
```

✅ GOOD
```ruby
# user.rb
class User; end

# user_profile.rb
class UserProfile; end
```

### 1.2 Rails Directory Structure
Follow Rails' "convention over configuration" for project structure.

❌ BAD
```ruby
# app/services/user_management/create_user.rb
# app/validators/email_validator.rb
```

✅ GOOD
```ruby
# app/services/user_management/create_user.rb (for complex logic)
# app/validators/email_validator.rb (standard Rails convention)
```

### 1.3 Configuration Files
Place custom initialization code in `config/initializers`. Keep gem-specific initializers in separate files named after the gem.

❌ BAD
```ruby
# config/initializers/app_setup.rb
# Contains setup for CarrierWave, Sidekiq, and custom app config.
```

✅ GOOD
```ruby
# config/initializers/carrierwave.rb
# config/initializers/sidekiq.rb
# config/initializers/custom_app_config.rb
```

## 2. Formatting

Consistency is paramount. RuboCop enforces these rules automatically.

### 2.1 Indentation & Line Length
Use two-space soft tabs. Limit lines to 120 characters to balance readability and screen real estate.

❌ BAD
```ruby
def long_method_name(param1, param2, param3, param4, param5, param6, param7) # > 120 chars
    puts "This line is indented with four spaces."
end
```

✅ GOOD
```ruby
def long_method_name(param1, param2, param3, param4, param5, param6, param7)
  puts "This line is indented with two spaces."
end
```

### 2.2 Trailing Whitespace & Newlines
Remove all trailing whitespace. End every file with a single newline.

❌ BAD
```ruby
puts "Hello"  # Trailing spaces
# No newline at EOF
```

✅ GOOD
```ruby
puts "Hello"
# Single newline at EOF
```

## 3. Naming Conventions

Strictly follow Ruby community standards for naming.

### 3.1 Variables, Methods, Files
Use `snake_case`. Do not separate numbers from letters.

❌ BAD
```ruby
someVar = 1
def someMethod; end
HelloWorld.rb
```

✅ GOOD
```ruby
some_var = 1
def some_method; end
hello_world.rb
```

### 3.2 Classes & Modules
Use `CamelCase`.

❌ BAD
```ruby
class Some_Class; end
```

✅ GOOD
```ruby
class SomeClass; end
```

### 3.3 Constants
Use `SCREAMING_SNAKE_CASE`. Freeze mutable constants.

❌ BAD
```ruby
SomeConst = 5
ALL_COLORS = ['red', 'blue']
```

✅ GOOD
```ruby
SOME_CONST = 5
ALL_COLORS = ['RED', 'BLUE'].freeze
```

### 3.4 Predicate & Bang Methods
Predicate methods (return boolean) end with `?`. Methods with side effects or that raise exceptions end with `!`.

❌ BAD
```ruby
def is_valid; end
def save_record; end # if it raises on failure
```

✅ GOOD
```ruby
def valid?; end
def save_record!; end # raises ActiveRecord::RecordInvalid on failure
```

## 4. Common Patterns & Anti-patterns

Embrace idiomatic Ruby and Rails patterns.

### 4.1 Block Shorthand
Use `&:method_name` for simple blocks.

❌ BAD
```ruby
names.map { |name| name.upcase }
```

✅ GOOD
```ruby
names.map(&:upcase)
```

### 4.2 Conditional Guards
Prefer guard clauses to reduce nesting.

❌ BAD
```ruby
def process_data(data)
  if data
    if data.valid?
      # ... process ...
    else
      log_error("Invalid data")
    end
  else
    log_error("No data provided")
  end
end
```

✅ GOOD
```ruby
def process_data(data)
  return log_error("No data provided") unless data
  return log_error("Invalid data") unless data.valid?

  # ... process ...
end
```

### 4.3 Keyword Arguments
Use keyword arguments for clarity and flexibility, especially for methods with many parameters.

❌ BAD
```ruby
def create_user(name, email, role = 'guest')
  # ...
end
create_user('Alice', 'alice@example.com', 'admin')
```

✅ GOOD
```ruby
def create_user(name:, email:, role: 'guest')
  # ...
end
create_user(name: 'Alice', email: 'alice@example.com', role: 'admin')
```

### 4.4 Argument Forwarding
Use `...` for forwarding arguments in Ruby 3.0+.

❌ BAD
```ruby
def wrapper(*args, **kwargs, &block)
  target_method(*args, **kwargs, &block)
end
```

✅ GOOD
```ruby
def wrapper(...)
  target_method(...)
end
```

## 5. Performance Considerations

Write efficient code, especially when dealing with collections or database queries.

### 5.1 Eager Loading (Rails)
Prevent N+1 query issues by eager loading associations.

❌ BAD
```ruby
users = User.all
users.each { |user| puts user.posts.count } # N+1 query
```

✅ GOOD
```ruby
users = User.includes(:posts).all
users.each { |user| puts user.posts.count } # Single query for posts
```

### 5.2 Memoization
Cache expensive computation results within a method call.

❌ BAD
```ruby
def expensive_calculation
  # performs complex calculation every time
  @result = some_complex_operation
end
```

✅ GOOD
```ruby
def expensive_calculation
  @result ||= some_complex_operation
end
```

## 6. Security Best Practices

Prioritize security in all code.

### 6.1 Strong Parameters (Rails)
Always use strong parameters to prevent mass assignment vulnerabilities.

❌ BAD
```ruby
def update
  @user.update(params[:user]) # Mass assignment vulnerability
end
```

✅ GOOD
```ruby
def update
  @user.update(user_params)
end

private

def user_params
  params.require(:user).permit(:name, :email, :password)
end
```

### 6.2 SQL Injection Prevention
Use ActiveRecord's query methods or parameter binding. Never interpolate user input directly into SQL.

❌ BAD
```ruby
User.where("name = '#{params[:name]}'") # SQL injection risk
```

✅ GOOD
```ruby
User.where(name: params[:name])
User.where("name = ?", params[:name])
```

## 7. Error Handling

Handle errors gracefully and informatively.

### 7.1 Custom Exceptions
Define custom exceptions for specific error conditions.

❌ BAD
```ruby
raise "User not found"
```

✅ GOOD
```ruby
class UserNotFoundError < StandardError; end

# ...
raise UserNotFoundError, "User with ID #{id} not found"
```

### 7.2 `rescue` Blocks
Use `rescue` for expected error scenarios. Be specific about the exception type.

❌ BAD
```ruby
begin
  # ... risky operation ...
rescue
  puts "An error occurred!" # Catches all exceptions, hides bugs
end
```

✅ GOOD
```ruby
begin
  # ... risky operation ...
rescue ActiveRecord::RecordNotFound => e
  Rails.logger.warn("Record not found: #{e.message}")
  nil # or handle gracefully
rescue StandardError => e
  Rails.logger.error("Unexpected error: #{e.message}")
  raise # Re-raise if unhandled
end
```

## 8. API Design (Rails)

Build RESTful and consistent APIs.

### 8.1 RESTful Routes
Use Rails' `resources` for standard RESTful actions. Use `member` and `collection` for non-standard actions.

❌ BAD
```ruby
get '/users/:id/deactivate', to: 'users#deactivate'
get '/users/active', to: 'users#active'
```

✅ GOOD
```ruby
resources :users do
  member do
    post :deactivate
  end
  collection do
    get :active
  end
end
```

### 8.2 JSON API Standards
Return consistent JSON responses, often following JSON:API specification.

❌ BAD
```json
{ "user": { "id": 1, "name": "Alice" } }
```

✅ GOOD
```json
{
  "data": {
    "id": "1",
    "type": "users",
    "attributes": {
      "name": "Alice"
    }
  }
}
```

## 9. Testing Approaches

RSpec is our preferred testing framework.

### 9.1 RSpec for All Tests
Write unit, integration, and feature tests using RSpec.

❌ BAD
```ruby
# Some tests in Minitest, others in RSpec
```

✅ GOOD
```ruby
# spec/models/user_spec.rb (unit)
# spec/requests/users_spec.rb (integration/feature)
```

### 9.2 Clear Test Descriptions
Use descriptive `describe` and `it` blocks.

❌ BAD
```ruby
describe User do
  it 'works' do
    # ...
  end
end
```

✅ GOOD
```ruby
describe User, '#full_name' do
  it 'returns the first and last name concatenated' do
    user = build(:user, first_name: 'John', last_name: 'Doe')
    expect(user.full_name).to eq('John Doe')
  end
end
```

### 9.3 FactoryBot for Test Data
Use FactoryBot for creating test data, avoiding direct `create` calls in tests.

❌ BAD
```ruby
User.create!(name: 'Test User', email: 'test@example.com')
```

✅ GOOD
```ruby
create(:user) # Creates a user with default attributes from factory
build(:user, email: 'custom@example.com') # Builds without saving
```