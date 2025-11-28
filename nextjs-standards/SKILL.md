---
name: nextjs-standards
description: Enforce Next.js App Router best practices, Server Components usage, and project structure. Use when creating new pages, components, or data fetching logic in Next.js.
allowed-tools: Read, Grep, Glob
---

# Next.js App Router Standards

**IMPORTANT:** Always respond in Korean to the user.

## When to use
- **New Page/Component:** When adding new routes or UI elements.
- **Data Fetching:** When implementing API calls or database queries.
- **Optimization:** When improving LCP/FCP or SEO.

## Instructions

### 1. Server vs Client Components
1.1. **Decision Matrix:**
   - **Need SEO?** -> Server Component.
   - **Need Database Access?** -> Server Component.
   - **Need `useState` / `useEffect`?** -> Client Component.
   - **Need `onClick` / `onChange`?** -> Client Component.
   - **Need Browser API (localStorage, window)?** -> Client Component.

1.2. **Server Components (Default):**
   - **DEFAULT** for all components.
   - Use for data fetching, sensitive logic (API keys), and static content.
   - **NEVER** import Server Components into Client Components directly (pass as `children` prop if needed).

1.3. **Client Components:**
   - Add `'use client'` at the very top.
   - Keep them as **Leaf Nodes** (at the bottom of the tree) to minimize the client bundle.
   - Don't make the entire Page a Client Component if only a button needs interactivity.

### 2. Data Fetching Patterns
2.1. **Fetch in Server Components:**
   - Use `async/await` directly in the component.
   - **Parallel Fetching:** Use `Promise.all` to prevent waterfalls.
     ```ts
     const [user, posts] = await Promise.all([getUser(), getPosts()]);
     ```

2.2. **Caching Strategy:**
   - **Static (Default):** `fetch(url)` (caches indefinitely).
   - **ISR:** `fetch(url, { next: { revalidate: 60 } })`.
   - **Dynamic:** `fetch(url, { cache: 'no-store' })` (or use `export const dynamic = 'force-dynamic'`).

2.3. **Server Actions (Mutations):**
   - Use for form submissions and data updates.
   - Define in `actions.ts` with `'use server'`.
   - **Validation:** ALWAYS validate input using **Zod** before processing.

### 3. Project Structure & Organization
3.1. **Directory Structure:**
   - `app/`: Routes (page.tsx, layout.tsx).
   - `components/`: Reusable UI.
   - `lib/`: Business logic, DB clients, Utils.
   - `actions/`: Server Actions.

3.2. **Colocation:**
   - Keep styles (`*.module.css`) and tests (`*.test.tsx`) next to the component.

### 4. Optimization
4.1. **Images & Fonts:**
   - **MANDATORY:** Use `<Image />` and `next/font`.
   - Specify `sizes` prop for responsive images.

4.2. **Metadata:**
   - Use `generateMetadata` for dynamic SEO tags.

### Checklist
Before finishing, verify:
- [ ] `'use client'` is NOT on the root Page (unless absolutely necessary).
- [ ] Database calls happen in Server Components.
- [ ] Inputs in Server Actions are validated (Zod).
- [ ] Images use `next/image`.
- [ ] No waterfalls in data fetching (use Promise.all).

## Examples

### Bad vs Good: Client Component Usage

**BAD:**
```tsx
// app/page.tsx
'use client'; // Making the whole page client-side just for a button
import { useState } from 'react';

export default function Page() {
  const [count, setCount] = useState(0);
  return (
    <div>
      <h1>Static Content</h1>
      <button onClick={() => setCount(count+1)}>{count}</button>
    </div>
  );
}
```

**GOOD:**
```tsx
// app/page.tsx (Server Component)
import Counter from '@/components/Counter';

export default function Page() {
  return (
    <div>
      <h1>Static Content</h1>
      <Counter /> {/* Only this part is client-side */}
    </div>
  );
}
```

### Server Action with Zod Validation
```ts
// app/actions.ts
'use server';

import { z } from 'zod';

const schema = z.object({
  email: z.string().email(),
});

export async function subscribe(formData: FormData) {
  const validated = schema.safeParse({
    email: formData.get('email'),
  });

  if (!validated.success) {
    return { error: validated.error.flatten() };
  }

  // Save to DB...
  return { success: true };
}
```
