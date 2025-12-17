---
description: This guide provides opinionated, actionable best practices for building robust, scalable, and maintainable NestJS applications using TypeScript, emphasizing modern patterns and common pitfalls.
globs: **/*.{ts,tsx}
---
# NestJS Best Practices

NestJS is a powerful framework for building scalable Node.js applications. Adhering to these guidelines ensures your projects are maintainable, performant, and secure.

## 1. Code Organization and Structure

Adopt a feature-based, semi-hexagonal architecture. Each business domain gets its own module.

### ✅ GOOD: Feature-based Modules
Organize code by feature, not by type.

```typescript
// src/modules/users/users.module.ts
@Module({
  controllers: [UsersController],
  providers: [UsersService, UsersRepository],
  exports: [UsersService],
})
export class UsersModule {}
// src/modules/users/controllers/users.controller.ts
// src/modules/users/services/users.service.ts
// src/modules/users/repositories/users.repository.ts
// src/modules/users/dtos/create-user.dto.ts
```

### ❌ BAD: Type-based Flat Structure
Avoid scattering related logic across different top-level folders.

```
// src/controllers/users.controller.ts
// src/services/users.service.ts
```

### Naming Conventions
Follow standard TypeScript conventions for clarity:
*   **Files/Folders**: `kebab-case` (e.g., `user-profile.module.ts`)
*   **Classes**: `PascalCase` (e.g., `UsersService`, `CreateUserDto`)
*   **Interfaces**: `IPascalCase` (e.g., `IUserRepository`)
*   **Functions/Methods**: `camelCase` (e.g., `createUser`, `findAllUsers`)
*   **Constants**: `UPPER_SNAKE_CASE` (e.g., `JWT_SECRET`)

## 2. Type Safety and Strictness

Always use explicit types. Avoid `any`. Enable strict TypeScript in `tsconfig.json`.

### ✅ GOOD: Explicit Types
Every variable, parameter, and return value must have an explicit type.

```typescript
interface IUser { id: string; name: string; email: string; }
async function fetchUserById(id: string): Promise<IUser | null> {
  const user: IUser | undefined = await this.userRepository.findOne(id);
  return user ?? null;
}
```

### ❌ BAD: Implicit `any`
Leads to runtime errors and defeats TypeScript's purpose.

```typescript
function fetchUserById(id) { // id is implicitly 'any'
  const user = this.userRepository.findOne(id);
  return user;
}
```

## 3. Common Patterns

### Dependency Injection (DI)
Leverage NestJS's DI system for loose coupling and testability.

### ✅ GOOD: Constructor Injection
Always inject dependencies via the constructor.

```typescript
@Injectable()
export class UsersService {
  constructor(private readonly userRepository: IUserRepository) {}
  async findAll(): Promise<IUser[]> { return this.userRepository.findAll(); }
}
```

### ❌ BAD: Manual Instantiation
Avoid `new` keyword for services/providers.

```typescript
const userService = new UsersService(); // Breaks DI and testability
```

### Data Transfer Objects (DTOs)
Use DTOs with `class-validator` for all incoming request bodies and query parameters.

### ✅ GOOD: Validated DTOs
Ensure data integrity at the edge of your application.

```typescript
// src/modules/users/dtos/create-user.dto.ts
import { IsEmail, IsString, MinLength } from 'class-validator';
export class CreateUserDto {
  @IsString() @MinLength(3) name: string;
  @IsEmail() email: string;
  @IsString() @MinLength(8) password: string;
}
// src/modules/users/controllers/users.controller.ts
@Post()
async createUser(@Body() createUserDto: CreateUserDto): Promise<IUser> {
  return this.userService.create(createUserDto);
}
```

### ❌ BAD: Direct Request Body Usage
Unvalidated data is a security risk.

```typescript
@Post()
async createUser(@Body() body: any): Promise<IUser> { /* ... */ }
```

### Controllers and Services
Controllers handle HTTP requests and delegate business logic to services. Services contain the core business logic and interact with data layers.

### ✅ GOOD: Lean Controllers, Rich Services
Controllers should be thin, focusing on request/response.

```typescript
// src/modules/users/controllers/users.controller.ts
@Controller('users')
export class UsersController {
  constructor(private readonly usersService: UsersService) {}
  @Get(':id')
  async findOne(@Param('id') id: string): Promise<IUser> {
    return this.usersService.findById(id);
  }
}
// src/modules/users/services/users.service.ts
@Injectable()
export class UsersService {
  constructor(private readonly userRepository: IUserRepository) {}
  async findById(id: string): Promise<IUser> {
    const user = await this.userRepository.findOne(id);
    if (!user) { throw new NotFoundException(`User ID "${id}" not found.`); }
    return user;
  }
}
```

### ❌ BAD: Business Logic in Controllers
Makes controllers hard to test and re-use.

```typescript
@Controller('users')
export class UsersController {
  constructor(private readonly userRepository: IUserRepository) {}
  @Get(':id')
  async findOne(@Param('id') id: string): Promise<IUser> {
    const user = await this.userRepository.findOne(id); // Logic here
    if (!user) { throw new NotFoundException(`User ID "${id}" not found.`); }
    return user;
  }
}
```

## 4. Error Handling

Centralize error handling using Exception Filters.

### ✅ GOOD: Global Exception Filters
Catch and format all unhandled exceptions consistently.

```typescript
// src/shared/common/filters/http-exception.filter.ts
@Catch(HttpException)
export class HttpExceptionFilter implements ExceptionFilter {
  catch(exception: HttpException, host: ArgumentsHost) {
    const ctx = host.switchToHttp(); const response = ctx.getResponse<Response>();
    const request = ctx.getRequest<Request>(); const status = exception.getStatus();
    response.status(status).json({
      statusCode: status, timestamp: new Date().toISOString(), path: request.url,
      message: exception.message,
    });
  }
}
// src/main.ts
async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  app.useGlobalFilters(new HttpExceptionFilter()); // Apply globally
  await app.listen(3000);
}
```

### ❌ BAD: `try/catch` in Every Method
Leads to boilerplate and inconsistent error responses.

```typescript
@Get(':id')
async findOne(@Param('id') id: string): Promise<IUser> {
  try { return await this.usersService.findById(id); }
  catch (error) { throw new HttpException('Failed', HttpStatus.INTERNAL_SERVER_ERROR); }
}
```

## 5. Security Best Practices

### Guards for Authorization
Use `Guards` for declarative route protection.

### ✅ GOOD: Role-Based Access Control (RBAC)
```typescript
// src/modules/auth/guards/roles.guard.ts
@Injectable()
export class RolesGuard implements CanActivate {
  constructor(private reflector: Reflector) {}
  canActivate(context: ExecutionContext): boolean {
    const requiredRoles = this.reflector.getAllAndOverride<UserRole[]>(ROLES_KEY, [
      context.getHandler(), context.getClass(),
    ]);
    if (!requiredRoles) { return true; }
    const { user } = context.switchToHttp().getRequest();
    return requiredRoles.some((role) => user.roles?.includes(role));
  }
}
// src/modules/users/controllers/users.controller.ts
@Controller('users')
@UseGuards(JwtAuthGuard, RolesGuard) // Apply guards
export class UsersController {
  @Get() @Roles(UserRole.Admin) // Require Admin role
  async findAll(): Promise<IUser[]> { return this.usersService.findAll(); }
}
```

### Environment Variables
Manage configuration securely using `@nestjs/config`.

### ✅ GOOD: `ConfigModule`
```typescript
// src/app.module.ts
@Module({
  imports: [ConfigModule.forRoot({ isGlobal: true, envFilePath: `.env.${process.env.NODE_ENV}` })],
})
export class AppModule {}
// In a service
@Injectable()
export class DatabaseService {
  constructor(private configService: ConfigService) {
    const dbHost = this.configService.get<string>('DATABASE_HOST');
  }
}
```

### ❌ BAD: Hardcoding Secrets
Never hardcode sensitive information.

```typescript
const JWT_SECRET = 'super-secret-key'; // BAD
```

## 6. Performance Considerations

### Fastify Adapter
Use Fastify for improved performance over Express.

### ✅ GOOD: Fastify Integration
```typescript
// src/main.ts
async function bootstrap() {
  const app = await NestFactory.create<NestFastifyApplication>(
    AppModule, new FastifyAdapter() // Use Fastify
  );
  await app.listen(3000);
}
bootstrap();
```

### Caching
Implement caching for frequently accessed, slow-changing data.

### ✅ GOOD: `CacheModule`
```typescript
// src/app.module.ts
@Module({
  imports: [CacheModule.register({ store: redisStore, host: 'localhost', port: 6379, ttl: 300 })],
})
export class AppModule {}
// In a service method
@Get() @UseInterceptors(CacheInterceptor) @CacheKey('all_users') @CacheTTL(60)
async findAll(): Promise<IUser[]> { return this.usersService.findAll(); }
```

## 7. Testing

Prioritize unit tests for services and e2e tests for controllers.

### ✅ GOOD: Unit Testing Services
Mock external