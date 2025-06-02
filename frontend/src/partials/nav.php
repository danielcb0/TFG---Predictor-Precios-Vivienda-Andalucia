<?php

// --------------------------------------
// Menú de navegación principal
// --------------------------------------

// Asegurarnos de tener la sesión iniciada para mostrar enlaces según el usuario
if (session_status() === PHP_SESSION_NONE) {
    session_start();
}
?>
<nav class="site-nav">
  <ul class="nav-list">
    <li class="nav-item"><a class="nav-link" href="index.php">Inicio</a></li>
    <li class="nav-item"><a class="nav-link" href="../pages/salas.php">Aulas</a></li>

    
    <?php if (!empty($_SESSION['user'])): ?>
      <li class="nav-item"><a class="nav-link" href="../pages/reservas.php">Mis Reservas</a></li>
      <li class="nav-item"><a class="nav-link" href="../pages/
      perfil.php"><?= htmlspecialchars($_SESSION['user']['name']) ?></a></li>

      
      <?php if (isset($_SESSION['user']['role']) && $_SESSION['user']['role'] === 'admin'): ?>
        <li class="nav-item"><a class="nav-link" href="../admin/dashboard.php">Panel Admin</a></li>
      <?php endif; ?>

      <li class="nav-item"><a class="nav-link" href="../pages/logout.php">Cerrar Sesión</a></li>
    <?php else: ?>
      <li class="nav-item"><a class="nav-link" href="../pages/login.php">Acceder</a></li>
      <li class="nav-item"><a class="nav-link" href="../pages/registro.php">Registrarse</a></li>
    <?php endif; ?>
  </ul>
</nav>
