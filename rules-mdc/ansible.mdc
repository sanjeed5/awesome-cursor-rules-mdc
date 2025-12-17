---
description: This guide defines definitive best practices for writing maintainable, performant, and robust Ansible code, focusing on modern YAML hygiene, role design, and CI/CD integration.
globs: **/*
---
# ansible Best Practices

Ansible excels when treated as infrastructure-as-code. These guidelines ensure your playbooks, roles, and modules are consistent, idempotent, and production-ready.

## 1. Code Organization and Structure

**Always structure your Ansible content into roles and collections.** Roles encapsulate specific functionalities, while collections group related roles, modules, and plugins, promoting reusability and discoverability.

-   **Role Scoping:** Each role or module must focus on a single, well-defined task. Avoid monolithic roles.
    ❌ BAD:
    ```yaml
    # roles/webserver/tasks/main.yml
    - name: Install Nginx
      ansible.builtin.apt: name=nginx state=present
    - name: Configure Nginx
      ansible.builtin.template: src=nginx.conf.j2 dest=/etc/nginx/nginx.conf
    - name: Start Nginx
      ansible.builtin.service: name=nginx state=started
    - name: Install PHP-FPM
      ansible.builtin.apt: name=php-fpm state=present
    # ... and more
    ```
    ✅ GOOD:
    ```yaml
    # roles/nginx/tasks/main.yml
    - name: Ensure Nginx is installed and configured
      ansible.builtin.include_tasks: setup.yml
    - name: Ensure Nginx service is running
      ansible.builtin.service: name=nginx state=started enabled=true

    # roles/php-fpm/tasks/main.yml
    - name: Ensure PHP-FPM is installed and configured
      ansible.builtin.include_tasks: setup.yml
    ```
-   **Fact Modules:** Create dedicated `_info` or `_facts` modules/tasks for gathering information, rather than overloading existing ones.
    ✅ GOOD:
    ```yaml
    # roles/system_facts/tasks/main.yml
    - name: Gather custom system facts
      ansible.builtin.set_fact:
        custom_os_version: "{{ ansible_facts['distribution_major_version'] }}"
    ```

## 2. YAML Hygiene and Syntax

**Enforce strict YAML formatting.** Consistent syntax improves readability and prevents subtle parsing errors.

-   **File Start:** All YAML files must begin with `---`.
-   **Indentation:** Use two spaces for indentation, never tabs.
-   **Quotes:** Quote strings only when necessary (e.g., when they contain special characters or Jinja map references). Prefer single quotes. Use double quotes for escaping characters or multi-line strings.
    ❌ BAD:
    ```yaml
    - name: "Install a package"
      ansible.builtin.apt: name="my-package" state="present"
    - name: Set fact
      set_fact:
        myvar: '{{ item["key"] }}' # Incorrect quoting for Jinja map
    ```
    ✅ GOOD:
    ```yaml
    ---
    - name: Install a package
      ansible.builtin.apt:
        name: my-package
        state: present
    - name: Set fact
      ansible.builtin.set_fact:
        my_var: "{{ item['key'] }}" # Double quotes for Jinja map
    - name: Print multi-line message
      ansible.builtin.debug:
        msg: |
          This is line one.
          This is line two.
    ```
-   **Colon Spacing:** Use exactly one space after a colon in key-value pairs.
    ❌ BAD:
    ```yaml
    - name : start service
      service:
          name    : my_service
          state   : started
    ```
    ✅ GOOD:
    ```yaml
    - name: start service
      ansible.builtin.service:
        name: my_service
        state: started
    ```
-   **Structured Map Style:** Always use the structured map style for module parameters, not the legacy `key=value` style.
    ❌ BAD:
    ```yaml
    - name: Create directory
      ansible.builtin.file: path=/opt/app state=directory mode=0755
    ```
    ✅ GOOD:
    ```yaml
    - name: Create directory
      ansible.builtin.file:
        path: /opt/app
        state: directory
        mode: '0755' # Quote octal modes
    ```
-   **Booleans:** Standardize on `true`/`false`.
    ❌ BAD:
    ```yaml
    - name: Enable service
      ansible.builtin.service: name=my_service enabled=yes
      become: True
    ```
    ✅ GOOD:
    ```yaml
    - name: Enable service
      ansible.builtin.service: name=my_service enabled=true
      become: true
    ```

## 3. Naming Conventions

**Adopt consistent `snake_case` naming.** This applies to variables, tasks, files, and role names. Prefix role names with the collection namespace to prevent collisions.

-   **Variables, Tasks, Files:**
    ❌ BAD:
    ```yaml
    # vars/main.yml
    myVariable: "value"
    My_Task_Name: "Install Software"
    ```
    ✅ GOOD:
    ```yaml
    # vars/main.yml
    my_variable: "value"
    # tasks/install_software.yml
    - name: Install software
      ansible.builtin.apt: name=software state=present
    ```
-   **Role Names:**
    ✅ GOOD: `my_namespace.my_collection.webserver`

## 4. Idempotency and Declarative Style

**Write idempotent tasks.** Running a playbook multiple times should yield the same result without unintended side effects. Favor declarative modules over imperative `command` or `shell`.

-   **Declarative Modules:**
    ❌ BAD:
    ```yaml
    - name: Check if file exists and create if not
      ansible.builtin.shell: |
        if [ ! -f /etc/myapp.conf ]; then
          cp /etc/myapp.conf.template /etc/myapp.conf
        fi
      args:
        creates: /etc/myapp.conf # Still imperative logic
    ```
    ✅ GOOD:
    ```yaml
    - name: Ensure myapp.conf exists
      ansible.builtin.copy:
        src: myapp.conf.template
        dest: /etc/myapp.conf
        owner: root
        group: root
        mode: '0644'
        backup: true # Idempotent file management
    ```
-   **Atomic File Operations:** Always use `atomic_move` (implicitly via `copy`, `template` modules) for file modifications to prevent data corruption.

## 5. Configuration Management

**Manage configurations with templates and structured variables.**

-   **Templates:** Use Jinja2 templates (`template` module) for dynamic configuration files.
-   **Variables:** Define role variables in `defaults/main.yml` (overridable) and `vars/main.yml` (not overridable by inventory/extra-vars). Prefer inventory variables for host-specific data.
    ❌ BAD:
    ```yaml
    # playbook.yml
    - hosts: webservers
      vars:
        nginx_port: 8080 # Hardcoded in playbook
      tasks:
        - name: Configure Nginx
          ansible.builtin.template: src=nginx.conf.j2 dest=/etc/nginx/nginx.conf
    ```
    ✅ GOOD:
    ```yaml
    # roles/nginx/defaults/main.yml
    nginx_port: 80 # Default port

    # inventory/group_vars/webservers.yml
    nginx_port: 8080 # Override default for webservers group

    # roles/nginx/templates/nginx.conf.j2
    listen {{ nginx_port }};
    ```

## 6. Performance Considerations

**Optimize playbook execution for speed.**

-   **Fact Gathering:** Disable fact gathering (`gather_facts: false`) if not explicitly needed. Use `setup` or `ansible.builtin.gather_facts` module selectively.
    ❌ BAD:
    ```yaml
    - hosts: all
      # gather_facts: true by default, even if not used
      tasks:
        - name: Ping all hosts
          ansible.builtin.ping:
    ```
    ✅ GOOD:
    ```yaml
    - hosts: all
      gather_facts: false # No facts needed for ping
      tasks:
        - name: Ping all hosts
          ansible.builtin.ping:
    ```
-   **Parallelism:** Leverage Ansible's parallelism by default, but be mindful of resource constraints.

## 7. Error Handling and Logging

**Implement robust error handling and informative logging.**

-   **Fail Gracefully:** Use `fail_json` (in custom modules) or `ansible.builtin.fail` for clear error messages. Avoid raw tracebacks.
    ✅ GOOD:
    ```yaml
    - name: Ensure critical variable is defined
      ansible.builtin.fail:
        msg: "Required variable 'app_version' is not defined."
      when: app_version is not defined
    ```
-   **Debug Verbosity:** Use the `ansible.builtin.debug` module with verbosity levels for controlled output.
    ✅ GOOD:
    ```yaml
    - name: Show debug info
      ansible.builtin.debug:
        var: my_variable
      when: ansible_verbosity >= 2 # Only show with -vv or higher
    ```

## 8. Testing and Linting

**Integrate linting and testing into your development workflow.**

-   **Ansible Lint:** Run `ansible-lint` on every pull request. It enforces the official style guide and detects common issues.
-   **Ansible Test:** Use `ansible-test` for unit and integration testing of custom modules and roles.
    ✅ GOOD:
    ```bash
    # In your CI pipeline or pre-commit hook
    ansible-lint .
    ansible-test sanity --test flake8 --test ansible-doc --test pep8 # For custom modules
    ```

## 9. CI/CD Integration

**Embed Ansible into your automated CI/CD pipelines.**

-   **Git Triggers:** Trigger playbook runs on Git pushes to automatically provision or update infrastructure.
-   **Collections:** Store reusable roles and modules in Ansible Collections (Galaxy or private registries) for easy dependency management.
-   **Pipeline Checks:** Include `ansible-lint` and collection dependency checks in your CI pipeline.
    ✅ GOOD:
    ```yaml
    # .github/workflows/ansible.yml
    name: Ansible CI/CD

    on: [push, pull_request]

    jobs:
      lint:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - name: Install Ansible and dependencies
            run: pip install ansible ansible-lint
          - name: Run Ansible Lint
            run: ansible-lint .

      deploy:
        runs-on: ubuntu-latest
        needs: lint
        if: github.ref == 'refs/heads/main'
        steps:
          - uses: actions/checkout@v4
          - name: Install Ansible Collections
            run: ansible-galaxy collection install -r requirements.yml
          - name: Run Playbook
            env:
              ANSIBLE_VAULT_PASSWORD: ${{ secrets.ANSIBLE_VAULT_PASSWORD }}
            run: ansible-playbook -i inventory/prod.yml site.yml --vault-password-file /dev/stdin
    ```