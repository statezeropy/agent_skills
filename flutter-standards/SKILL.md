---
name: flutter-standards
description: Enforce Flutter & Dart coding standards, Effective Dart guidelines, and performance best practices. Use when writing Flutter code, refactoring widgets, or reviewing Dart files.
allowed-tools: Read, Grep, Glob
---

# Flutter & Dart Standards

**IMPORTANT:** Always respond in Korean to the user.

## When to use
- **New Feature Development:** When creating new widgets or screens.
- **Refactoring:** When optimizing existing code for performance or readability.
- **Code Review:** When checking for anti-patterns or style violations.

## Instructions

### 1. Dart Style & Effective Dart
1.1. **Naming Conventions:**
   - **Types (Classes, Enums):** `UpperCamelCase`.
   - **Variables, Functions, Constants:** `lowerCamelCase`.
   - **Files, Directories, Packages:** `snake_case` (lowercase_with_underscores).
   - **Private Members:** Prefix with `_`.

1.2. **Null Safety:**
   - Explicitly mark nullable types (`String?`).
   - Use `late` only when necessary and safe.
   - **NEVER** use `!` (bang operator) unless you have checked for null immediately before. Prefer `?` or `??`.

1.3. **Variables:**
   - Prefer `final` for variables that don't change.
   - Prefer `const` for compile-time constants (especially Widgets).

1.4. **Async:**
   - Use `async`/`await` over raw `Future.then()`.
   - Handle errors with `try-catch`.

### 2. Flutter Best Practices
2.1. **Widget Composition:**
   - Break large widgets into smaller, reusable widgets.
   - Prefer `StatelessWidget` if state is not needed.
   - **PERFORMANCE:** Use `const` constructors for Widgets whenever possible. This allows Flutter to cache the widget.

2.2. **State Management Decision Tree:**
   - **Is it UI state only affecting this widget?** (e.g., expanded/collapsed, text input) -> Use `setState`.
   - **Is it shared across multiple widgets?** (e.g., User session, Cart) -> Use **Riverpod** (or Provider/Bloc as per project config).
   - **Avoid Global Variables** for state.

2.3. **Build Method:**
   - Keep `build()` methods pure and free of side effects (no API calls).
   - Move complex logic out of `build()`.

2.4. **Performance Optimization:**
   - **Lists:** Use `ListView.builder` for long or infinite lists. NEVER use `ListView(children: ...)` for dynamic data.
   - **Rebuilds:** Use `Consumer` (Riverpod) or `Selector` (Provider) to listen to specific parts of the state, minimizing rebuilds.

### 3. Project Structure
3.1. **Standard Layout:**
   - `lib/main.dart`: Entry point.
   - `lib/src/`: Private implementation details.
   - `lib/features/`: Feature-based organization (e.g., `auth/`, `profile/`).
     - `presentation/`: Widgets, Pages.
     - `domain/`: Models, Entities.
     - `data/`: Repositories, Data Sources.

3.2. **Assets:**
   - Define assets in `pubspec.yaml`.
   - Use a generated class (e.g., `Assets.gen.dart`) or constants for asset paths to avoid typos.

### Checklist
Before finishing, verify:
- [ ] File names are `snake_case.dart`.
- [ ] Class names are `UpperCamelCase`.
- [ ] `const` used for Widgets where possible.
- [ ] No logic in `build()` method.
- [ ] `ListView.builder` used for lists.
- [ ] `flutter_lints` passing.

## Examples

### Bad vs Good: Widget Structure

**BAD:**
```dart
// Too big, no const, logic in build
class MyScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    var data = fetchData(); // Side effect in build!
    return Scaffold(
      body: Column(
        children: [
          Text("Title"), // Missing const
          // ... 100 lines of nested widgets
        ],
      ),
    );
  }
}
```

**GOOD:**
```dart
class MyScreen extends StatelessWidget {
  const MyScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      body: Column(
        children: [
          HeaderWidget(title: "Title"),
          UserListWidget(),
        ],
      ),
    );
  }
}
```

### Async Operation & Error Handling
```dart
Future<void> loadData() async {
  try {
    final data = await apiService.fetch();
    // process data
  } catch (e, stackTrace) {
    // Log error properly
    logger.e("Failed to load data", error: e, stackTrace: stackTrace);
    // Show user feedback
  }
}
```

### Performance: ListView
```dart
// Efficient for long lists
ListView.builder(
  itemCount: items.length,
  itemBuilder: (context, index) {
    return UserListItem(user: items[index]);
  },
)
```
