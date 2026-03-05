import { test, expect } from '@playwright/test';

test.describe('SEG Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the dashboard
    await page.goto('/');
  });

  test('should display the dashboard title', async ({ page }) => {
    await expect(page.getByRole('heading', { name: /Smart Gallery/i })).toBeVisible();
  });

  test('should have person upload section', async ({ page }) => {
    await expect(page.getByText(/Person to search/i)).toBeVisible();
  });

  test('should have upload all images section', async ({ page }) => {
    await expect(page.getByText(/Upload all images/i)).toBeVisible();
  });

  test('should have matched images section', async ({ page }) => {
    await expect(page.getByText(/Matched images/i)).toBeVisible();
  });
});
