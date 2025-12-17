---
description: This guide provides definitive best practices for Android SDK development in Java, focusing on modern architecture, Jetpack libraries, performance, security, and testing.
globs: **/*.java
---
# android-sdk Best Practices (Java)

This guide outlines the essential practices for building robust, performant, and maintainable Android applications using Java, adhering to modern standards and leveraging the AndroidX ecosystem.

## 1. Code Organization and Structure

**1.1 Modular Architecture (MVVM)**
Adopt a clean, modular architecture, with MVVM (Model-View-ViewModel) as the standard. This promotes separation of concerns, testability, and scalability.

❌ **BAD: Tight Coupling**
```java
// Activity directly fetching data and updating UI
public class UserProfileActivity extends AppCompatActivity {
    private TextView userName;
    // ... other UI elements

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_user_profile);
        userName = findViewById(R.id.user_name);
        // Direct network call on UI thread (or poorly managed background thread)
        new Thread(() -> {
            String name = fetchUserNameFromNetwork(); // Blocking call
            runOnUiThread(() -> userName.setText(name));
        }).start();
    }
    // ...
}
```

✅ **GOOD: MVVM with Repository Pattern**
```java
// ui/UserProfileActivity.java
public class UserProfileActivity extends AppCompatActivity {
    private UserProfileViewModel viewModel;
    private TextView userName;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_user_profile);
        userName = findViewById(R.id.user_name);

        viewModel = new ViewModelProvider(this).get(UserProfileViewModel.class);
        viewModel.getUserName().observe(this, name -> {
            userName.setText(name);
        });
    }
}

// ui/UserProfileViewModel.java
public class UserProfileViewModel extends AndroidViewModel {
    private UserRepository userRepository;
    private MutableLiveData<String> userName = new MutableLiveData<>();

    public UserProfileViewModel(@NonNull Application application) {
        super(application);
        this.userRepository = new UserRepository(application); // Injected via Hilt in real app
        loadUserName();
    }

    public LiveData<String> getUserName() {
        return userName;
    }

    private void loadUserName() {
        // Use ExecutorService or RxJava for background work
        Executors.newSingleThreadExecutor().execute(() -> {
            String name = userRepository.fetchUserName();
            userName.postValue(name); // Post value to main thread
        });
    }
}

// data/UserRepository.java
public class UserRepository {
    // In a real app, this would interact with Room, Retrofit, etc.
    public String fetchUserName() {
        // Simulate network call
        try { Thread.sleep(2000); } catch (InterruptedException e) {}
        return "John Doe";
    }
}
```

**1.2 Package Structure**
Organize packages by feature or layer, not by Android component type.

❌ **BAD: Component-based**
```
com.example.app
├── activities
│   └── UserProfileActivity.java
├── fragments
│   └── SettingsFragment.java
├── adapters
│   └── UserListAdapter.java
└── utils
    └── NetworkUtils.java
```

✅ **GOOD: Feature-based (or Layer-based for larger apps)**
```
com.example.app
├── userprofile
│   ├── UserProfileActivity.java
│   ├── UserProfileViewModel.java
│   └── UserProfileFragment.java
├── settings
│   ├── SettingsActivity.java
│   └── SettingsViewModel.java
├── data
│   ├── UserRepository.java
│   ├── UserApiService.java
│   └── UserDatabase.java
└── common
    ├── adapters
    │   └── UserListAdapter.java
    └── utils
        └── NetworkUtils.java
```

## 2. Common Patterns and Anti-patterns

**2.1 Jetpack Libraries**
Always prefer AndroidX Jetpack libraries for modern, backward-compatible, and lifecycle-aware components.

*   **Room for Persistence:** Use Room for all local database storage.
    ❌ **BAD: Raw SQLiteOpenHelper**
    ```java
    // Manual SQLite table creation and query management
    public class MyDbHelper extends SQLiteOpenHelper { /* ... */ }
    ```
    ✅ **GOOD: Room Database**
    ```java
    @Database(entities = {User.class}, version = 1)
    public abstract class AppDatabase extends RoomDatabase {
        public abstract UserDao userDao();
    }

    @Entity
    public class User {
        @PrimaryKey public int id;
        public String name;
    }

    @Dao
    public interface UserDao {
        @Query("SELECT * FROM user WHERE id = :userId")
        LiveData<User> getUserById(int userId);

        @Insert
        void insert(User user);
    }
    ```

*   **View Binding (or Data Binding):** Eliminate `findViewById` boilerplate.
    ❌ **BAD: Manual View Lookup**
    ```java
    public class MyActivity extends AppCompatActivity {
        private Button myButton;
        @Override protected void onCreate(Bundle savedInstanceState) {
            super.onCreate(savedInstanceState);
            setContentView(R.layout.activity_my);
            myButton = findViewById(R.id.my_button);
            myButton.setOnClickListener(v -> /* ... */);
        }
    }
    ```
    ✅ **GOOD: View Binding**
    ```java
    public class MyActivity extends AppCompatActivity {
        private ActivityMyBinding binding; // Generated binding class
        @Override protected void onCreate(Bundle savedInstanceState) {
            super.onCreate(savedInstanceState);
            binding = ActivityMyBinding.inflate(getLayoutInflater());
            setContentView(binding.getRoot());
            binding.myButton.setOnClickListener(v -> /* ... */);
        }
    }
    ```

*   **Navigation Component:** Manage in-app navigation consistently.
    ❌ **BAD: Manual Fragment Transactions**
    ```java
    getSupportFragmentManager().beginTransaction()
        .replace(R.id.fragment_container, new DetailFragment())
        .addToBackStack(null)
        .commit();
    ```
    ✅ **GOOD: Navigation Graph**
    ```java
    // In Activity/Fragment
    NavController navController = Navigation.findNavController(this, R.id.nav_host_fragment);
    navController.navigate(R.id.action_homeFragment_to_detailFragment);
    // Define navigation in res/navigation/nav_graph.xml
    ```

**2.2 Dependency Injection (Hilt)**
Use Hilt (built on Dagger) for robust dependency injection, simplifying object creation and improving testability.

❌ **BAD: Manual Dependency Management**
```java
public class MyRepository {
    private MyApiService apiService = new MyApiService(); // Hardcoded dependency
    // ...
}
```
✅ **GOOD: Hilt**
```java
@Module
@InstallIn(SingletonComponent.class)
public abstract class AppModule {
    @Provides
    @Singleton
    public static MyApiService provideMyApiService() {
        return new MyApiService(); // Or provide Retrofit instance
    }
}

@Singleton
public class MyRepository {
    private final MyApiService apiService;

    @Inject // Hilt injects MyApiService
    public MyRepository(MyApiService apiService) {
        this.apiService = apiService;
    }
    // ...
}

@HiltAndroidApp
public class MyApplication extends Application {}
```

## 3. Performance Considerations

**3.1 Layout Optimization**
Flatten view hierarchies using `ConstraintLayout` and `ViewStub` to reduce layout inflation time and memory usage.

❌ **BAD: Deeply Nested Layouts**
```xml
<LinearLayout orientation="vertical">
    <LinearLayout orientation="horizontal">
        <TextView />
        <ImageView />
    </LinearLayout>
    <LinearLayout orientation="vertical">
        <TextView />
        <Button />
    </LinearLayout>
</LinearLayout>
```
✅ **GOOD: ConstraintLayout**
```xml
<androidx.constraintlayout.widget.ConstraintLayout>
    <TextView app:layout_constraintTop_toTopOf="parent" />
    <ImageView app:layout_constraintBaseline_toBaselineOf="@id/textView" />
    <TextView app:layout_constraintTop_toBottomOf="@id/imageView" />
    <Button app:layout_constraintTop_toBottomOf="@id/textView2" />
</androidx.constraintlayout.widget.ConstraintLayout>
```

**3.2 Image Loading**
Always use a dedicated image loading library like Glide for efficient image fetching, caching, and display.

❌ **BAD: Manual Bitmap Loading**
```java
// Prone to OOM errors, poor caching, complex lifecycle management
Bitmap bitmap = BitmapFactory.decodeStream(urlConnection.getInputStream());
imageView.setImageBitmap(bitmap);
```
✅ **GOOD: Glide**
```java
Glide.with(context)
     .load("https://example.com/image.jpg")
     .placeholder(R.drawable.placeholder)
     .error(R.drawable.error)
     .into(imageView);
```

**3.3 Background Processing**
Use WorkManager for deferrable, guaranteed background tasks. For immediate, short-lived tasks, use `ExecutorService`.

❌ **BAD: Inconsistent Background Tasks**
```java
// Using AlarmManager for non-exact tasks, or custom Services for simple work
new AsyncTask<Void, Void, Void>() { /* ... */ }.execute(); // Deprecated
```
✅ **GOOD: WorkManager**
```java
// Define a WorkRequest
OneTimeWorkRequest uploadWorkRequest =
    new OneTimeWorkRequest.Builder(UploadWorker.class)
        .setConstraints(new Constraints.Builder()
            .setRequiredNetworkType(NetworkType.CONNECTED)
            .build())
        .build();

// Enqueue the work
WorkManager.getInstance(context).enqueue(uploadWorkRequest);

// Your Worker class
public class UploadWorker extends Worker {
    public UploadWorker(@NonNull Context context, @NonNull WorkerParameters workerParams) {
        super(context, workerParams);
    }

    @NonNull
    @Override
    public Result doWork() {
        // Perform long-running task
        return Result.success();
    }
}
```

## 4. Common Pitfalls and Gotchas

**4.1 Context Leaks**
Avoid holding strong references to `Activity` or `Fragment` contexts in long-lived objects. Use `ApplicationContext` when a long-lived context is truly needed, or weak references.

❌ **BAD: Static Context Reference**
```java
public class BadManager {
    private static Context context; // LEAK!
    public static void init(Context ctx) { context = ctx; }
}
```
✅ **GOOD: Application Context or No Context**
```java
public class GoodManager {
    private Context appContext; // Use application context for long-lived needs
    public GoodManager(Context context) {
        this.appContext = context.getApplicationContext();
    }
    // Or, better yet, pass only what's needed, not the whole context
    public void doSomething(String data) { /* ... */ }
}
```

**4.2 Security Best Practices**
Implement Android's security guidelines for inter-app communication and network security.

*   **Explicit Intents with Choosers:** For sensitive implicit intents.
    ❌ **BAD: Implicit Intent for Sensitive Data**
    ```java
    Intent sendIntent = new Intent(Intent.ACTION_SEND);
    sendIntent.putExtra(Intent.EXTRA_TEXT, "Sensitive data");
    sendIntent.setType("text/plain");
    startActivity(sendIntent); // Any app can intercept
    ```
    ✅ **GOOD: App Chooser**
    ```java
    Intent sendIntent = new Intent(Intent.ACTION_SEND);
    sendIntent.putExtra(Intent.EXTRA_TEXT, "Sensitive data");
    sendIntent.setType("text/plain");
    Intent chooser = Intent.createChooser(sendIntent, getResources().getString(R.string.chooser_title));
    startActivity(chooser);
    ```

*   **Non-Exported Content Providers:** Prevent unauthorized access.
    ❌ **BAD: Exported Content Provider (default on older Android)**
    ```xml
    <provider android:name=".MyContentProvider" android:authorities="com.example.app.provider" />
    ```
    ✅ **GOOD: Non-Exported Content Provider**
    ```xml
    <provider android:name=".MyContentProvider" android:authorities="com.example.app.provider" android:exported="false" />
    ```

*   **Network Security Configuration:** Enforce HTTPS and manage trust anchors.
    ❌ **BAD: Allowing Cleartext Traffic**
    ```xml
    <network-security-config>
        <base-config cleartextTrafficPermitted="true">
            <trust-anchors>
                <certificates src="system" />
            </trust-anchors>
        </base-config>
    </network-security-config>
    ```
    ✅ **GOOD: Disabling Cleartext Traffic**
    ```xml
    <!-- res/xml/network_security_config.xml -->
    <network-security-config>
        <domain-config cleartextTrafficPermitted="false">
            <domain includeSubdomains="true">api.example.com</domain>
            <domain includeSubdomains="true">secure.example.com</domain>
        </domain-config>
        <!-- Use debug-overrides for development only -->
        <debug-overrides>
            <trust-anchors>
                <certificates src="user"/>
            </trust-anchors>
        </debug-overrides>
    </network-security-config>
    ```
    And in `AndroidManifest.xml`:
    ```xml
    <application android:networkSecurityConfig="@xml/network_security_config" ...>
    ```

**4.3 Ignoring Lint Warnings**
Treat all lint warnings as errors. Android Studio's lint tool catches critical issues from performance to security.

❌ **BAD: Shipping with Lint Warnings**
```
> Task :app:lintDebug
...
W: /path/to/project/app/src/main/java/com/example/app/MainActivity.java:123: Warning: Hardcoded string "Hello" should use @string resource [HardcodedText]
```
✅ **GOOD: Resolve All Lint Issues**
Configure Gradle to fail builds on lint errors:
```gradle
android {
    lintOptions {
        abortOnError true // Fail the build if lint finds errors
        warningsAsErrors true // Treat warnings as errors
    }
}
```

## 5. Testing Approaches

**5.1 Unit Tests**
Focus on testing business logic, ViewModels, and Repositories in isolation on the JVM. Use JUnit and Mockito.

```java
// test/com/example/app/userprofile/UserProfileViewModelTest.java
public class UserProfileViewModelTest {
    @Rule
    public InstantTaskExecutorRule instantTaskExecutorRule = new InstantTaskExecutorRule(); // For LiveData

    @Mock
    private UserRepository userRepository;
    @Mock
    private Application application; // Mock application context

    private UserProfileViewModel viewModel;

    @Before
    public void setup() {
        MockitoAnnotations.initMocks(this);
        viewModel = new UserProfileViewModel(application);
        // Inject mock repository (in real app, Hilt would handle this)
        ReflectionHelpers.setField(viewModel, "userRepository", userRepository);
    }

    @Test
    public void getUserName_returnsCorrectName() {
        // Given
        when(userRepository.fetchUserName()).thenReturn("Test User");

        // When
        viewModel.loadUserName(); // Trigger data load

        // Then
        assertEquals("Test User", viewModel.getUserName().getValue());
    }
}
```

**5.2 Integration Tests**
Test interactions between components (e.g., Repository with Room database, or API service with Retrofit). Run on an Android emulator or device.

```java
// androidTest/com/example/app/data/UserDaoTest.java
@RunWith(AndroidJUnit4.class)
public class UserDaoTest {
    @Rule
    public InstantTaskExecutorRule instantTaskExecutorRule = new InstantTaskExecutorRule(); // For LiveData

    private AppDatabase database;
    private UserDao userDao;

    @Before
    public void createDb() {
        Context context = ApplicationProvider.getApplicationContext();
        database = Room.inMemoryDatabaseBuilder(context, AppDatabase.class).build();
        userDao = database.userDao();
    }

    @After
    public void closeDb() {
        database.close();
    }

    @Test
    public void writeUserAndReadInList() {
        User user = new User();
        user.id = 1;
        user.name = "Test User";
        userDao.insert(user);

        User byId = LiveDataTestUtil.getValue(userDao.getUserById(1));
        assertEquals(user.name, byId.name);
    }
}
```

**5.3 UI Tests (Espresso)**
Verify user flows and UI correctness. Run on an Android emulator or device.

```java
// androidTest/com/example/app/UserProfileActivityTest.java
@RunWith(AndroidJUnit4.class)
@LargeTest
public class UserProfileActivityTest {
    @Rule
    public ActivityScenarioRule<UserProfileActivity> activityRule =
            new ActivityScenarioRule<>(UserProfileActivity.class);

    @Test
    public void userNameIsDisplayed() {
        // Assuming the ViewModel loads "John Doe"
        onView(withId(R.id.user_name))
            .check(matches(withText("John Doe")));
    }
}
```

**5.4 Automated Testing**
Integrate all tests (unit, integration, UI, lint) into your CI/CD pipeline.

```bash
# Run lint checks
./gradlew lintDebug

# Run unit tests
./gradlew testDebugUnitTest

# Run connected (integration/UI) tests on connected device/emulator
./gradlew connectedDebugAndroidTest
```